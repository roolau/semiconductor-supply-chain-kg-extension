"""Three clustering views on the final Phase 1 graph (design doc §8.1). No extraction, no LLM.

  product    what a company makes: Jaccard distance on its product categories (products.py, strict), agglomerative
             (average linkage), cut at a distance threshold. Only companies with a product profile.
  ecosystem  who trades with whom: Louvain communities (networkx) on the undirected SUPPLIES_TO graph, edges weighted
             by fused confidence. Feeds the recommender's peer sets. (Design doc says Leiden; Louvain is built into
             networkx, so no extra install; same idea.)
  role       where a company sits in the chain, in two steps:
             (1) companies with a product profile: the product group with the most evidence (claims per item edge),
                 chain groups (materials / equipment / design tools-IP / chips / manufacturing) first when they hold
                 >= 25% of it - NVIDIA's software catalogue must not make it a software company;
             (2) everyone else: label propagation on the supply graph to a tier (upstream / chips / downstream):
                 one tier above its customers, or one below its suppliers.
             Graded against CSET ETO (eto.py); baselines: Phase 1 BELONGS_TO_STAGE labels, and an unsupervised
             Ward clustering of role features (kept as a comparison: it does much worse).
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from .eto import ETO
from .load import KG
from .products import Products

GROUP = {
    "raw_materials": "materials", "wafers_substrates": "materials", "process_chemicals": "materials",
    "industrial_specialty_gases": "materials",
    "fab_equipment": "equipment", "test_equipment": "equipment", "equipment_subsystems": "equipment",
    "eda_ip": "design_tools_ip",
    "logic_chips": "chips", "memory_chips": "chips", "analog_power_chips": "chips",
    "foundry_services": "manufacturing", "packaging_osat": "manufacturing", "ems_odm_services": "manufacturing",
    "pcb_boards": "components", "components_passive_connectors": "components", "optical_components": "components",
    "servers_systems": "systems", "networking_equipment": "systems", "storage": "systems", "devices": "systems",
    "compute_capacity": "cloud_software", "cloud_services": "cloud_software", "software_licenses": "cloud_software",
    "other_off_chain": "other",
}
GROUPS = ["materials", "equipment", "design_tools_ip", "chips", "manufacturing", "components", "systems",
          "cloud_software", "other"]
TO_ETO_ROLE = {"chips": "chip_design", "design_tools_ip": "design_tools_ip", "equipment": "equipment",
               "materials": "materials"}  # other groups have no ETO counterpart
STAGE_OF_CATEGORY = {"eda_ip": "Design", "logic_chips": "Design", "memory_chips": "Design", "analog_power_chips": "Design",
                     "fab_equipment": "Fabrication", "equipment_subsystems": "Fabrication", "foundry_services": "Fabrication",
                     "raw_materials": "Fabrication", "wafers_substrates": "Fabrication", "process_chemicals": "Fabrication",
                     "industrial_specialty_gases": "Fabrication", "packaging_osat": "ATP", "test_equipment": "ATP"}
CONF_W = {"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.4}
CHAIN_GROUPS = ["materials", "equipment", "design_tools_ip", "chips", "manufacturing"]
TIER_OF_GROUP = {"materials": 0, "equipment": 0, "design_tools_ip": 1, "chips": 1, "manufacturing": 1,
                 "components": 1, "systems": 2, "cloud_software": 2, "other": 2}
TIER_NAME = {0: "upstream", 1: "chips", 2: "downstream"}
ETO_TIER = {"materials": "upstream", "equipment": "upstream", "chip_design": "chips", "design_tools_ip": "chips"}
# value-chain lanes (CSIS "Simplified Depiction of the Semiconductor Value Chain", extended downstream to AI systems)
LANE_OF_CATEGORY = {
    "eda_ip": "design_sw_ip", "logic_chips": "chips", "memory_chips": "chips", "analog_power_chips": "chips",
    "raw_materials": "materials", "wafers_substrates": "materials", "process_chemicals": "materials",
    "industrial_specialty_gases": "materials", "fab_equipment": "equipment", "equipment_subsystems": "equipment",
    "test_equipment": "equipment", "foundry_services": "fabrication", "packaging_osat": "assembly_test",
    "ems_odm_services": "components_ems", "pcb_boards": "components_ems", "components_passive_connectors": "components_ems",
    "optical_components": "components_ems", "servers_systems": "systems_cloud", "networking_equipment": "systems_cloud",
    "storage": "systems_cloud", "devices": "systems_cloud", "compute_capacity": "systems_cloud",
    "cloud_services": "systems_cloud", "software_licenses": "systems_cloud"}
LANES = {"chips": "Chips (design / IDM)", "fabrication": "Fabrication (foundries)", "assembly_test": "Assembly & test",
         "systems_cloud": "Systems, cloud & devices", "design_sw_ip": "Design software & IP",
         "materials": "Materials & chemicals", "equipment": "Manufacturing equipment",
         "unknown_inputs": "Suppliers, products unknown", "components_ems": "Components & EMS"}


def group_shares(cats: set[str]) -> np.ndarray:
    c = Counter(GROUP[x] for x in cats if x in GROUP)
    v = np.array([c[g] for g in GROUPS], dtype=float)
    return v / v.sum() if v.sum() else v


class Clusters:
    def __init__(self, kg: KG, products: Products | None = None, supply: set | None = None):
        self.kg = kg
        self.P = products or Products(kg)
        self.supply = set(supply) if supply is not None else kg.supply_edges()  # pass a training graph to avoid leakage
        self.companies = sorted({x for e in self.supply for x in e} | {c for c in self.P.makes_items if kg.is_company(c)})

    # ---------------- product ----------------
    def product(self, threshold: float = 0.5) -> dict[str, int]:
        ids = [c for c in self.companies if self.P.makes(c)]
        if len(ids) < 2:
            return {c: 0 for c in ids}
        cats = sorted({x for c in ids for x in self.P.makes(c)})
        X = np.array([[x in self.P.makes(c) for x in cats] for c in ids], dtype=bool)
        lab = fcluster(linkage(pdist(X, "jaccard"), "average"), t=threshold, criterion="distance")
        return dict(zip(ids, (int(x) for x in lab)))

    # ---------------- ecosystem ----------------
    def graph(self, supply=None) -> nx.Graph:
        supply = supply if supply is not None else self.supply
        conf = {(e["subject"], e["object"]): CONF_W.get(e.get("confidence"), 0.7)
                for e in self.kg.by_relation.get("SUPPLIES_TO", [])}
        G = nx.Graph()
        for a, b in sorted(supply):  # sorted: Louvain with a fixed seed is only reproducible in a fixed edge order
            w = conf.get((a, b), 0.7)
            G.add_edge(a, b, weight=max(w, G.get_edge_data(a, b, {}).get("weight", 0)))
        return G

    def ecosystem(self, resolution: float = 1.0, seed: int = 42, supply=None) -> dict[str, int]:
        G = self.graph(supply)
        comms = nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
        comms = sorted(comms, key=len, reverse=True)
        return {c: i for i, cs in enumerate(comms) for c in cs}

    # ---------------- role ----------------
    def role_features(self) -> tuple[list[str], np.ndarray]:
        out_n, in_n = defaultdict(set), defaultdict(set)
        for a, b in self.supply:
            out_n[a].add(b)
            in_n[b].add(a)
        D = nx.DiGraph(list(self.supply))
        sinks = [n for n in D if D.out_degree(n) == 0]
        down = nx.multi_source_dijkstra_path_length(D.reverse(copy=False), sinks) if sinks else {}
        own = {c: group_shares(self.P.makes(c)) for c in self.companies}
        rows, ids = [], []
        for c in self.companies:
            cust = [own[x] for x in out_n[c] if own[x].sum()]
            supp = [own[x] for x in in_n[c] if own[x].sum()]
            cv = np.mean(cust, axis=0) if cust else np.zeros(len(GROUPS))
            sv = np.mean(supp, axis=0) if supp else np.zeros(len(GROUPS))
            o, i = len(out_n[c]), len(in_n[c])
            deg = [math.log1p(o), math.log1p(i), (o + 1) / (o + i + 2), min(down.get(c, 0), 5) / 5]
            if not (own[c].sum() or cv.sum() or sv.sum()):
                continue  # nothing to place it by
            ids.append(c)
            rows.append(np.concatenate([2.0 * own[c], cv, sv, deg]))  # own products weigh double
        return ids, np.array(rows)

    def role_unsupervised(self, k: int = 8) -> dict[str, dict]:
        ids, X = self.role_features()
        Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
        lab = fcluster(linkage(Z, "ward"), t=k, criterion="maxclust")
        g = len(GROUPS)
        names = {}
        for cl in set(lab):
            members = X[lab == cl]
            own_mean, cust_mean = members[:, :g].mean(0), members[:, g:2 * g].mean(0)
            if own_mean.sum() > 0.2:
                names[cl] = GROUPS[int(own_mean.argmax())]
            else:  # no products known: name by what its customers make ("supplier to chips")
                names[cl] = "supplier_to_" + GROUPS[int(cust_mean.argmax())] if cust_mean.sum() else "buyer"
        return {c: {"role_cluster": int(l), "role_name": names[l]} for c, l in zip(ids, lab)}

    def weighted_shares(self, c: str) -> np.ndarray:
        """Product-group shares weighted by evidence: claims behind each PROVIDES item edge."""
        w = self._claims()
        v = np.zeros(len(GROUPS))
        for i in self.P.makes_items.get(c, ()):
            cat = self.P.cat.get(i)
            if cat in GROUP:
                v[GROUPS.index(GROUP[cat])] += max(1, w.get((c, i), 1))
        return v / v.sum() if v.sum() else v

    def _claims(self) -> dict:
        if not hasattr(self, "_w"):
            self._w = {(e["subject"], e["object"]): len(e.get("claim_ids") or [])
                       for e in self.kg.by_relation.get("PROVIDES", [])}
        return self._w

    def role_by_products(self) -> dict[str, str]:
        out = {}
        chain = [GROUPS.index(g) for g in CHAIN_GROUPS]
        for c in self.companies:
            v = self.weighted_shares(c)
            if not v.sum():
                continue
            if v[chain].sum() >= 0.25:
                out[c] = GROUPS[chain[int(v[chain].argmax())]]
            else:
                out[c] = GROUPS[int(v.argmax())]
        return out

    def lanes(self) -> dict[str, dict]:
        """Value-chain lane per company. From products (evidence-weighted, chain lanes first when they hold >= 25%),
        else from the propagated tier: upstream -> 'unknown_inputs', downstream -> systems_cloud, chips -> chips."""
        w = self._claims()
        out = {}
        for c in self.companies:
            v = Counter()
            for i in sorted(self.P.makes_items.get(c, ())):
                lane = LANE_OF_CATEGORY.get(self.P.cat.get(i))
                if lane:
                    v[lane] += max(1, w.get((c, i), 1))
            if v:
                chain = {k: x for k, x in v.items() if k != "systems_cloud"}
                best = max(sorted(chain), key=chain.get) if chain and sum(chain.values()) / sum(v.values()) >= 0.25 \
                    else max(sorted(v), key=v.get)
                out[c] = {"lane": best, "placed_by": "products"}
        for c, t in self.tiers().items():
            if c not in out:
                out[c] = {"lane": {"upstream": "unknown_inputs", "downstream": "systems_cloud", "chips": "chips"}[t["tier"]],
                          "placed_by": t["tier_from"]}
        return out

    def lane_flows(self, lanes: dict | None = None) -> list[dict]:
        """Supply links counted between lanes (supplier lane -> customer lane), largest first."""
        lanes = lanes or self.lanes()
        n = Counter((lanes[a]["lane"], lanes[b]["lane"]) for a, b in self.supply if a in lanes and b in lanes)
        return [{"from": a, "to": b, "links": k} for (a, b), k in n.most_common()]

    def tiers(self, rounds: int = 5) -> dict[str, dict]:
        """Seed tiers from role_by_products, then propagate along SUPPLIES_TO to companies without a profile."""
        seed = {c: TIER_OF_GROUP[g] for c, g in self.role_by_products().items()}
        tier, how = dict(seed), {c: "products" for c in seed}
        cust, supp = defaultdict(set), defaultdict(set)
        for a, b in self.supply:
            cust[a].add(b)
            supp[b].add(a)
        for _ in range(rounds):
            new = {}
            for c in self.companies:
                if c in seed:
                    continue
                ct = [tier[x] for x in cust[c] if x in tier]
                st = [tier[x] for x in supp[c] if x in tier]
                if ct:
                    new[c] = (max(0, round(float(np.median(ct))) - 1), "below customers")
                elif st:
                    new[c] = (min(2, round(float(np.median(st))) + 1), "above suppliers")
            if all(tier.get(c) == t for c, (t, _) in new.items()):
                break
            for c, (t, h) in new.items():
                tier[c], how[c] = t, h
        return {c: {"tier": TIER_NAME[t], "tier_from": how[c]} for c, t in tier.items()}

    def role(self, k: int = 8) -> dict[str, dict]:
        """Comparison only: unsupervised Ward clustering of role features into k clusters."""
        return self.role_unsupervised(k)

    def stage_from_products(self) -> dict[str, str]:
        """Stage with the most product evidence (claims); ties broken in a fixed order, so runs agree."""
        out = {}
        for c in self.companies:
            st = Counter()
            for cat, w in sorted(self.P.makes_weighted(c).items()):
                if cat in STAGE_OF_CATEGORY:
                    st[STAGE_OF_CATEGORY[cat]] += w
            if st:
                out[c] = max(sorted(st), key=lambda k: st[k])
        return out

    def phase1_stage(self) -> dict[str, str]:
        """Baseline: Phase 1 LLM stage labels (item BELONGS_TO_STAGE stage), lifted to the companies providing the item."""
        item_stage = {e["subject"]: self.kg.name(e["object"]) for e in self.kg.by_relation.get("BELONGS_TO_STAGE", [])}
        norm = {"Design": "Design", "Fabrication": "Fabrication", "Assembly, testing, and packaging": "ATP"}
        per = defaultdict(Counter)
        for e in self.kg.by_relation.get("PROVIDES", []):
            if e["object"] in item_stage and self.kg.is_company(e["subject"]):
                per[e["subject"]][norm.get(item_stage[e["object"]], item_stage[e["object"]])] += 1
        return {c: v.most_common(1)[0][0] for c, v in per.items()}


def _acc(pred: dict, gold: dict, key: str, mapping=None) -> dict:
    graded = {c: g[key] for c, g in gold.items() if g.get(key)}
    got = {c: (mapping.get(pred[c]) if mapping else pred[c]) for c in graded if c in pred}
    hit = sum(got[c] == graded[c] for c in got)
    return {"graded": len(graded), "covered": len(got), "correct": hit,
            "accuracy_on_covered": round(hit / len(got), 3) if got else None,
            "accuracy_on_all": round(hit / len(graded), 3) if graded else None}


def evaluate(C: Clusters, unsup: dict, tiers: dict, eto: dict) -> dict:
    from sklearn.metrics import adjusted_rand_score
    both = [c for c in eto if c in unsup]
    eto_tier = {c: {"eto_tier": ETO_TIER[g["eto_role"]]} for c, g in eto.items()}
    return {
        "eto_matched_companies": len(eto),
        "role": {
            "rule_products": _acc(C.role_by_products(), eto, "eto_role", TO_ETO_ROLE),
            "unsupervised_ward_k": _acc({c: v["role_name"] for c, v in unsup.items()}, eto, "eto_role", TO_ETO_ROLE),
            "ari_unsupervised_vs_eto": round(adjusted_rand_score([eto[c]["eto_role"] for c in both],
                                                                 [unsup[c]["role_cluster"] for c in both]), 3) if both else None,
        },
        "tier": {
            "products_plus_propagation": _acc({c: v["tier"] for c, v in tiers.items()}, eto_tier, "eto_tier"),
            # split by how the tier was found. 16 Nov: "below customers" 6/6, "above suppliers" 0/6 - a company seen only
            # as a buyer (Tesla, Google, Honeywell) is placed by what it buys, while ETO grades what it sells.
            **{f"from_{h.replace(' ', '_')}": _acc({c: v["tier"] for c, v in tiers.items() if v["tier_from"] == h},
                                                   {c: eto_tier[c] for c in eto_tier if tiers.get(c, {}).get("tier_from") == h},
                                                   "eto_tier")
               for h in ("products", "below customers", "above suppliers")},
        },
        "stage": {
            "from_products": _acc(C.stage_from_products(), eto, "eto_stage"),
            "phase1_llm_stage_baseline": _acc(C.phase1_stage(), eto, "eto_stage"),
        },
    }


def disagreements(C: Clusters, prod: dict, eco: dict, top: int = 10) -> list[dict]:
    """Same product cluster, different ecosystems: the insight-slide examples, biggest companies first."""
    deg = Counter(x for e in C.supply for x in e)
    by = defaultdict(list)
    for c, p in prod.items():
        if c in eco:
            by[p].append(c)
    out = []
    for p, cs in by.items():
        cs = sorted(cs, key=lambda c: -deg[c])
        for i, a in enumerate(cs):
            for b in cs[i + 1:]:
                if eco[a] != eco[b]:
                    out.append({"a": C.kg.name(a), "b": C.kg.name(b), "product_cluster": p, "ecosystem_a": eco[a],
                                "ecosystem_b": eco[b], "makes_shared": sorted(C.P.makes(a) & C.P.makes(b)),
                                "degree": deg[a] + deg[b]})
    return sorted(out, key=lambda r: -r["degree"])[:top]


def build(kg: KG, out_dir, threshold: float = 0.5, resolution: float = 1.0, k: int = 8) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    C = Clusters(kg)
    prod, eco = C.product(threshold), C.ecosystem(resolution)
    rules, tiers, unsup = C.role_by_products(), C.tiers(), C.role_unsupervised(k)
    eto = ETO().match(kg)
    with open(out / "clusters.jsonl", "w", encoding="utf-8") as f:
        for c in C.companies:
            f.write(json.dumps({"company_id": c, "name": kg.name(c), "makes": sorted(C.P.makes(c)),
                                "product_cluster": prod.get(c), "ecosystem": eco.get(c),
                                "role": rules.get(c), **tiers.get(c, {"tier": None, "tier_from": None}),
                                "role_cluster_unsupervised": unsup.get(c, {}).get("role_cluster"),
                                **(eto.get(c) or {})}, ensure_ascii=False) + "\n")
    size = lambda d: sorted(Counter(d.values()).values(), reverse=True)
    report = {
        "params": {"product_threshold": threshold, "louvain_resolution": resolution, "role_k": k},
        "product": {"companies": len(prod), "clusters": len(set(prod.values())), "sizes": size(prod)},
        "ecosystem": {"companies": len(eco), "clusters": len(set(eco.values())), "sizes": size(eco)[:15],
                      "modularity": round(nx.community.modularity(C.graph(), [{c for c in eco if eco[c] == i}
                                                                               for i in set(eco.values())]), 3)},
        "role": {"with_product_role": len(rules), "roles": dict(Counter(rules.values()).most_common()),
                 "with_tier": len(tiers), "tiers": dict(Counter(v["tier"] for v in tiers.values())),
                 "tier_from": dict(Counter(v["tier_from"] for v in tiers.values()))},
        "evaluation": evaluate(C, unsup, tiers, eto),
        "disagreements": disagreements(C, prod, eco),
    }
    (out / "clustering_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report
