"""Relationship recommendation on the final Phase 1 graph: who could become a company's next customer (or supplier).

Scorers, each fitted on a TRAINING graph only (held-out edges never touch peers, ecosystems, stages or products):
  popularity   how many suppliers a candidate already buys from                    (baseline)
  peers        customer-overlap collaborative filtering: companies that share customers with you; what else they
               supply                                                               ("similar suppliers, similar customers")
  product      product peers: companies that make what you make; their customers
  stage        P(customer stage | your stage) from stage-to-stage supply links     (value-chain view)
  ecosystem    1 if the candidate is in your Louvain community
  hybrid       reciprocal-rank fusion of the scorers above, then the soft product gate
Product gate: fail -> dropped, unknown -> kept (products.py). Every recommendation carries its evidence paths.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .categories import label
from .cluster import LANES, Clusters
from .load import KG
from .products import Products

RRF_K = 10  # reciprocal-rank fusion constant: small, so a top-3 rank from one scorer counts


class Recommender:
    def __init__(self, kg: KG, supply: set[tuple[str, str]], products: Products | None = None,
                 direction: str = "customers", peer_product: bool = True):
        """direction='customers': recommend customers for a supplier; 'suppliers': the reverse."""
        self.kg, self.direction, self.peer_product = kg, direction, peer_product
        self.P = products or Products(kg)
        self.train = set(supply)
        # oriented view: "out" = the side we recommend
        edges = self.train if direction == "customers" else {(b, a) for a, b in self.train}
        self.out, self.inn = defaultdict(set), defaultdict(set)
        for a, b in edges:
            self.out[a].add(b)
            self.inn[b].add(a)
        self.nodes = sorted({x for e in kg.supply_edges() for x in e})
        self.idx = {n: i for i, n in enumerate(self.nodes)}
        C = Clusters(kg, self.P, supply=self.train)
        self.eco = C.ecosystem(supply=self.train)
        self.lane = {c: v["lane"] for c, v in C.lanes().items()}
        flows = Counter((self.lane.get(a), self.lane.get(b)) for a, b in edges if a in self.lane and b in self.lane)
        tot = Counter()
        for (la, _), k in flows.items():
            tot[la] += k
        self.stage_p = {(la, lb): k / tot[la] for (la, lb), k in flows.items()}

    # ---------------- scorers: arrays over self.nodes ----------------
    def popularity(self, a: str) -> np.ndarray:
        return np.array([len(self.inn[n]) for n in self.nodes], dtype=float)

    def peers(self, a: str) -> np.ndarray:
        s = np.zeros(len(self.nodes))
        mine = self.out[a]
        if not mine:
            return s
        cands = {b for d in mine for b in self.inn[d]} - {a}
        for b in cands:
            sim = len(mine & self.out[b]) / math.sqrt(len(mine) * len(self.out[b])) * self.alike(a, b)
            for d in self.out[b]:
                s[self.idx[d]] += sim
        return s

    def alike(self, a: str, b: str) -> float:
        """How much a shared customer says: a lot between product peers, little between unlike suppliers
        (ASML and Qorvo both selling to Samsung says nothing about ASML selling to Apple)."""
        if not self.peer_product:
            return 1.0
        pa, pb = self.P.makes_primary(a), self.P.makes_primary(b)
        if not pa or not pb:
            return 0.6
        return 0.3 + 0.7 * len(pa & pb) / len(pa | pb)

    def product(self, a: str) -> np.ndarray:
        s = np.zeros(len(self.nodes))
        mk = self.P.makes_primary(a)
        if not mk:
            return s
        for b in self.out:
            if b == a:
                continue
            ob = self.P.makes_primary(b)
            if ob:
                sim = len(mk & ob) / len(mk | ob)
                if sim:
                    for d in self.out[b]:
                        s[self.idx[d]] += sim
        return s

    def stage(self, a: str) -> np.ndarray:
        la = self.lane.get(a)
        return np.array([self.stage_p.get((la, self.lane.get(n)), 0.0) for n in self.nodes])

    def ecosystem(self, a: str) -> np.ndarray:
        e = self.eco.get(a)
        return np.array([float(e is not None and self.eco.get(n) == e) for n in self.nodes])

    def gate(self, a: str) -> np.ndarray:
        """1 pass, 0.5 unknown, 0 fail (supplier makes nothing the customer buys)."""
        out = np.empty(len(self.nodes))
        for i, n in enumerate(self.nodes):
            sup, cus = (a, n) if self.direction == "customers" else (n, a)
            st, _ = self.P.gate_status(sup, cus, self.train)  # all categories: primary-only hurt (16 Nov grid)
            out[i] = {"pass": 1.0, "unknown": 0.5, "fail": 0.0}[st]
        return out

    SCORERS = ("popularity", "peers", "product", "stage", "ecosystem")
    # stage left out: 16 Nov ablation (5 seeds) hybrid MRR 0.258 with stage vs 0.248 without for customers, 0.310 vs
    # 0.313 for suppliers, and Hits@10 better without it in both. It stays in the explanations and the map.
    HYBRID = ("peers", "product", "popularity")

    def hybrid(self, a: str, parts=HYBRID) -> np.ndarray:
        s = np.zeros(len(self.nodes))
        for name in parts:
            v = getattr(self, name)(a)
            if not v.any():
                continue
            order = np.argsort(-v, kind="stable")
            rank = np.empty(len(v))
            rank[order] = np.arange(1, len(v) + 1)
            s += np.where(v > 0, 1.0 / (RRF_K + rank), 0.0)
        return s

    def score(self, a: str, method: str, gate: bool = False) -> np.ndarray:
        if method == "hybrid":
            s = self.hybrid(a, self.HYBRID)
        elif method.startswith("hybrid-"):  # ablation: hybrid without one scorer
            s = self.hybrid(a, tuple(x for x in self.HYBRID if x != method[7:]))
        elif method.startswith("hybrid_with_"):  # ablation: hybrid plus one more scorer
            s = self.hybrid(a, self.HYBRID + (method[12:],))
        else:
            s = getattr(self, method)(a)
        if gate:
            g = self.gate(a)
            s = np.where(g == 0, -1.0, s + 1e-6 * g)  # failed gate goes last; pass beats unknown on ties
        return s

    # ---------------- recommendations with evidence ----------------
    def recommend(self, a: str, k: int = 10, method: str = "hybrid") -> list[dict]:
        s = self.score(a, method, gate=True)
        known = self.out[a] | {a}
        order = [i for i in np.argsort(-s, kind="stable") if self.nodes[i] not in known and s[i] > 0][:k]
        return [{"company_id": self.nodes[i], "name": self.kg.name(self.nodes[i]), "score": round(float(s[i]), 4),
                 "evidence": self.explain(a, self.nodes[i])} for i in order]

    def evidence(self, a: str, d: str) -> dict:
        """The same evidence as explain(), as data (for the web page)."""
        peers = sorted(((len(self.out[a] & self.out[b]), b) for b in self.inn[d] if b != a and self.out[a] & self.out[b]),
                       reverse=True)[:2]
        sup, cus = (a, d) if self.direction == "customers" else (d, a)
        st, cats = self.P.gate_status(sup, cus, self.train)
        cats = (cats & self.P.makes_primary(sup)) or cats
        why = self.P.buys(cus, self.train) if st == "pass" else {}
        via = {c: [x.split(":", 1)[1] for x in why.get(c, []) if x.startswith("supplier:")][:2] for c in sorted(cats)}
        la, ld = self.lane.get(a), self.lane.get(d)
        return {"peers": [[b, sorted(self.out[a] & self.out[b])[:3]] for _, b in peers],
                "gate": st, "cats": sorted(cats), "via": via,
                "stage": [la, ld, round(self.stage_p.get((la, ld), 0.0), 3)] if la and ld else None,
                "eco": self.eco.get(a) is not None and self.eco.get(a) == self.eco.get(d)}

    def explain(self, a: str, d: str) -> list[str]:
        name = self.kg.name
        verb = "supplies" if self.direction == "customers" else "buys from"
        out = []
        peers = sorted(((len(self.out[a] & self.out[b]), b) for b in self.inn[d] if b != a and self.out[a] & self.out[b]),
                       reverse=True)[:2]
        for n_shared, b in peers:
            shared = sorted(self.out[a] & self.out[b], key=name)[:3]
            out.append(f"Peer: {name(b)} also {verb} {', '.join(map(name, shared))} (like you) and {verb} {name(d)}.")
        sup, cus = (a, d) if self.direction == "customers" else (d, a)
        st, cats = self.P.gate_status(sup, cus, self.train)
        cats = (cats & self.P.makes_primary(sup)) or cats  # name the supplier's main products when they fit
        if st == "pass":
            why = self.P.buys(cus, self.train)
            c = sorted(cats)[0]
            src = [x for x in why.get(c, []) if x.startswith("supplier:")][:2]
            via = f" (it buys {label(c)} from {', '.join(name(x.split(':', 1)[1]) for x in src)})" if src else ""
            out.append(f"Product fit: {name(sup)} makes {', '.join(label(x) for x in sorted(cats))}, "
                       f"which {name(cus)} buys{via}.")
        la, ld = self.lane.get(a), self.lane.get(d)
        if la and ld and self.stage_p.get((la, ld)):
            out.append(f"Stage: {LANES[la]} -> {LANES[ld]} carries {self.stage_p[(la, ld)]:.0%} of this stage's links.")
        if self.eco.get(a) is not None and self.eco.get(a) == self.eco.get(d):
            out.append(f"Same trading ecosystem ({self.eco[a]}).")
        return out or ["Popular buyer: no closer evidence path."]



# ---------------- evaluation: masked-edge holdout ----------------
def split(supply: set, frac: float = 0.2, seed: int = 0) -> tuple[set, set]:
    """Hide frac of the edges, only where both ends keep at least one other edge (no company vanishes)."""
    rng = random.Random(seed)
    deg = Counter(x for e in supply for x in e)
    edges = sorted(supply)
    rng.shuffle(edges)
    test, target = set(), int(frac * len(edges))
    for a, b in edges:
        if len(test) >= target:
            break
        if deg[a] > 1 and deg[b] > 1:
            test.add((a, b))
            deg[a] -= 1
            deg[b] -= 1
    return supply - test, test


def evaluate(kg: KG, seeds=(0, 1, 2, 3, 4), frac: float = 0.2, direction: str = "customers",
             methods=None) -> dict:
    """Filtered ranking: each held-out (A, D) is ranked among all companies that are not A's other known partners."""
    methods = methods or (["popularity", "peers", "product", "stage", "ecosystem"]
                          + ["hybrid"] + [f"hybrid-{x}" for x in Recommender.HYBRID]
                          + ["hybrid_with_stage", "hybrid_with_ecosystem"])
    P = Products(kg)
    supply = kg.supply_edges()
    res = defaultdict(lambda: defaultdict(list))
    per_lane = defaultdict(lambda: defaultdict(list))
    for seed in seeds:
        train, test = split(supply, frac, seed)
        R = Recommender(kg, train, P, direction)
        if direction == "suppliers":
            test = {(b, a) for a, b in test}
        all_pos = defaultdict(set)
        for a, b in (supply if direction == "customers" else {(y, x) for x, y in supply}):
            all_pos[a].add(b)
        by_a = defaultdict(list)
        for a, d in test:
            by_a[a].append(d)
        for a, ds in by_a.items():
            for m in methods:
                for g in (False, True):
                    s = R.score(a, m, gate=g)
                    for d in ds:
                        mask = np.array([n != a and (n == d or n not in all_pos[a]) for n in R.nodes])
                        sd = s[R.idx[d]]
                        rank = 1 + int(((s > sd) & mask).sum()) + int(((s == sd) & mask).sum() - 1) // 2  # ties: mean
                        key = m + ("+gate" if g else "")
                        res[key]["rr"].append(1 / rank)
                        res[key]["hit10"].append(rank <= 10)
                        if m == "hybrid" and g:
                            per_lane[R.lane.get(a, "none")]["hit10"].append(rank <= 10)
        res["_n"]["test_edges"].append(len(test))
    summary = {m: {"MRR": round(float(np.mean(v["rr"])), 3), "Hits@10": round(float(np.mean(v["hit10"])), 3),
                   "n": len(v["rr"])} for m, v in res.items() if not m.startswith("_")}
    summary = dict(sorted(summary.items(), key=lambda kv: -kv[1]["MRR"]))
    lanes = {LANES.get(l, l): {"Hits@10": round(float(np.mean(v["hit10"])), 3), "n": len(v["hit10"])}
             for l, v in per_lane.items()}
    return {"direction": direction, "seeds": list(seeds), "holdout_fraction": frac,
            "test_edges_per_seed": res["_n"]["test_edges"], "candidates": len(R.nodes),
            "methods": summary, "hybrid_gate_hits10_by_stage": lanes}


def build(kg: KG, out_dir, examples=("company_008", "company_002", "company_011")) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = (0, 1, 2, 3, 4)
    report = {"customers": evaluate(kg, seeds, direction="customers"),
              "suppliers": evaluate(kg, seeds, direction="suppliers")}
    full = Recommender(kg, kg.supply_edges(), direction="customers")
    report["examples"] = {kg.name(c): full.recommend(c, 5) for c in examples if c in full.idx}
    (out / "recommender_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report


# ---------------- competitors ----------------
class Competitors:
    """Who competes with a company: makes the same things and sells to the same customers. There is no COMPETES_WITH
    relation in the graph, so this is graded against ETO instead: two companies that ETO lists as providers of the same
    input count as competitors (evaluate_competitors).

      product    Jaccard similarity of the main product categories (products.makes_primary)
      items      number of identical PROVIDES items ("EUV lithography systems")
      customers  cosine similarity of their customer sets
      hybrid     reciprocal-rank fusion of the three
    """

    def __init__(self, kg: KG, supply: set | None = None, products: Products | None = None):
        self.kg, self.P = kg, products or Products(kg)
        self.supply = set(supply) if supply is not None else kg.supply_edges()
        self.out = defaultdict(set)
        for a, b in self.supply:
            self.out[a].add(b)
        self.nodes = sorted({x for e in kg.supply_edges() for x in e} | {c for c in self.P.makes_items if kg.is_company(c)})
        self.idx = {n: i for i, n in enumerate(self.nodes)}
        self.eco = Clusters(kg, self.P, supply=self.supply).ecosystem(supply=self.supply)

    def product(self, a, b):
        pa, pb = self.P.makes_primary(a), self.P.makes_primary(b)
        return len(pa & pb) / len(pa | pb) if pa and pb else 0.0

    def items(self, a, b):
        return float(len(self.P.makes_items.get(a, set()) & self.P.makes_items.get(b, set())))

    def customers(self, a, b):
        A, B = self.out[a], self.out[b]
        return len(A & B) / math.sqrt(len(A) * len(B)) if A and B else 0.0

    def popularity(self, a, b):
        return float(len(self.out[b]))

    def score(self, a: str, method: str = "hybrid") -> np.ndarray:
        parts = ("product", "items", "customers") if method == "hybrid" else (method,)
        s = np.zeros(len(self.nodes))
        for name in parts:
            f = getattr(self, name)
            v = np.array([f(a, b) if b != a else -1.0 for b in self.nodes])
            if method != "hybrid":
                return v
            order = np.argsort(-v, kind="stable")
            rank = np.empty(len(v))
            rank[order] = np.arange(1, len(v) + 1)
            s += np.where(v > 0, 1.0 / (RRF_K + rank), 0.0)
        return s

    def recommend(self, a: str, k: int = 8) -> list:
        if a not in self.idx:
            return []
        s = self.score(a)
        order = [i for i in np.argsort(-s, kind="stable") if self.nodes[i] != a and s[i] > 0][:k]
        return [[self.nodes[i], round(float(s[i]), 4), self.evidence(a, self.nodes[i])] for i in order]

    def evidence(self, a: str, b: str) -> dict:
        items = sorted(self.P.makes_items.get(a, set()) & self.P.makes_items.get(b, set()), key=self.kg.name)[:3]
        return {"cats": sorted(self.P.makes_primary(a) & self.P.makes_primary(b)),
                "items": [self.kg.name(i) for i in items],
                "customers": sorted(self.out[a] & self.out[b], key=self.kg.name)[:3],
                "eco": self.eco.get(a) is not None and self.eco.get(a) == self.eco.get(b)}


def evaluate_competitors(kg: KG) -> dict:
    """Against ETO: competitors of a company = graph companies that ETO lists as providers of at least one of the
    same inputs (photomasks left out). P@5 = share of the top 5 that are such competitors; Hit@10 = at least one in
    the top 10; MRR = 1 / rank of the first one. ETO is incomplete, so precision is a lower bound."""
    from .eto import CAPTIVE_ONLY, ETO
    from .names import norm_name
    E = ETO()
    index = {norm_name(o): o for o in E.provides}
    org = {}
    for eid, ent in kg.entities.items():
        if ent.get("entity_type") != "Company":
            continue
        for n in [ent.get("canonical_name")] + list(ent.get("aliases") or []):
            o = index.get(norm_name(n or ""))
            if o:
                org[eid] = o
                break
    inp = {c: E.provides[o] - CAPTIVE_ONLY for c, o in org.items()}
    gold = {a: {b for b in inp if b != a and inp[a] & inp[b]} for a in inp}
    C = Competitors(kg)
    gold = {a: {b for b in g if b in C.idx} for a, g in gold.items() if a in C.idx}
    gold = {a: g for a, g in gold.items() if g}
    res = {}
    for m in ("hybrid", "product", "items", "customers", "popularity"):
        p5, h10, rr = [], [], []
        for a, g in sorted(gold.items()):
            s = C.score(a, m)
            s[C.idx[a]] = -9
            order = [C.nodes[i] for i in np.argsort(-s, kind="stable")]
            p5.append(len(set(order[:5]) & g) / 5)
            first = min(order.index(b) + 1 for b in g)
            h10.append(first <= 10)
            rr.append(1 / first)
        res[m] = {"P@5": round(float(np.mean(p5)), 3), "Hit@10": round(float(np.mean(h10)), 3),
                  "MRR": round(float(np.mean(rr)), 3), "n": len(p5)}
    return {"companies": len(gold), "gold_pairs": sum(len(g) for g in gold.values()), "methods": res}
