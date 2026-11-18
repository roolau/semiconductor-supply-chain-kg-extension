"""Product-mediated relations, inferred from the Phase 1 graph (no extraction, no LLM).

  makes(A)  = categories of the items A PROVIDES
  buys(B)   = categories of the items B USES                                  (direct)
            + makes(S) for every company S that already SUPPLIES_TO B         (step 1: what B actually buys)
  Categories, not items, are matched                                          (step 2: "EUV photoresist" = "photoresist")

A pair A -> B passes the product gate when makes(A) and buys(B) share a category. Gate-passing pairs without an edge
are recommendation candidates (hypotheses), never written back as facts. Every function takes the supply-edge set
explicitly, so evaluation can recompute buys(B) without the held-out edge (no leakage).
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .categories import categorize
from .load import KG


def item_categories(kg: KG) -> dict[str, str | None]:
    items = {e["object"] for rel in ("PROVIDES", "USES") for e in kg.by_relation.get(rel, [])}
    return {i: categorize(kg.name(i)) for i in items}


def strong(e: dict) -> bool:
    """Evidence an item edge must have to count when strict=True: an SEC source, 2+ claims, or HIGH confidence.
    16 Nov spot check: the wrong ones (NVIDIA 'finFET logic process technologies', Micron 'central processing units')
    were all Wikipedia-only, 1-2 claims, MEDIUM; the right ones (ASML 'EUV lithography systems', Micron 'HBM') had SEC."""
    return "SEC" in (e.get("source_types") or []) or len(e.get("claim_ids") or []) >= 2 or e.get("confidence") == "HIGH"


class Products:
    def __init__(self, kg: KG, strict: bool = True):
        self.kg = kg
        self.strict = strict
        self.cat = item_categories(kg)
        self.makes_items: dict[str, set[str]] = defaultdict(set)
        self.uses_items: dict[str, set[str]] = defaultdict(set)
        for rel, target in (("PROVIDES", self.makes_items), ("USES", self.uses_items)):
            for e in kg.by_relation.get(rel, []):
                if kg.is_company(e["subject"]) and (not strict or strong(e)):
                    target[e["subject"]].add(e["object"])

    def makes_weighted(self, a: str) -> Counter:
        """Category -> evidence (claims behind each PROVIDES item edge)."""
        if not hasattr(self, "_claims"):
            self._claims = {(e["subject"], e["object"]): len(e.get("claim_ids") or [])
                            for e in self.kg.by_relation.get("PROVIDES", [])}
        w = Counter()
        for i in self.makes_items.get(a, ()):
            if self.cat.get(i):
                w[self.cat[i]] += max(1, self._claims.get((a, i), 1))
        return w

    def makes_primary(self, a: str, min_share: float = 0.15) -> set[str]:
        """Categories holding >= min_share of the company's product evidence: drops one-off items such as
        Lam Research 'semiconductor equipment engineering services' or ASML 'software and platforms'."""
        w = self.makes_weighted(a)
        tot = sum(w.values())
        return {c for c, x in w.items() if x / tot >= min_share} if tot else set()

    def _cats(self, items) -> set[str]:
        return {self.cat[i] for i in items if self.cat.get(i)}

    def makes(self, a: str) -> set[str]:
        return self._cats(self.makes_items.get(a, ()))

    def buys(self, b: str, supply: set[tuple[str, str]], via_suppliers: bool = True) -> dict[str, list[str]]:
        """{category: [why, ...]} - 'uses:<item>' or 'supplier:<company>'."""
        out: dict[str, list[str]] = defaultdict(list)
        for i in self.uses_items.get(b, ()):
            if self.cat.get(i):
                out[self.cat[i]].append(f"uses:{i}")
        if via_suppliers:
            for s, c in supply:
                if c == b:
                    for cat in self.makes(s):
                        out[cat].append(f"supplier:{s}")
        return dict(out)

    def gate(self, a: str, b: str, supply: set[tuple[str, str]], via_suppliers: bool = True) -> set[str]:
        """Shared categories between what A makes and what B buys (empty = gate fails)."""
        return self.makes(a) & set(self.buys(b, supply, via_suppliers))

    def coverage(self, supply: set[tuple[str, str]] | None = None) -> dict:
        """How well the gate explains EXISTING edges, leave-one-out: for each edge A->B, recompute buys(B) without it.
        Also how selective the gate is over all company pairs."""
        supply = supply if supply is not None else self.kg.supply_edges()
        companies = sorted({x for e in supply for x in e})
        res = {}
        for mode, via in (("uses_only_items", None), ("uses_only", False), ("uses_plus_suppliers", True)):
            hit = 0
            for a, b in supply:
                rest = supply - {(a, b)}
                if via is None:  # exact item match, the 'A provides x, B uses x' baseline
                    hit += bool(self.makes_items.get(a, set()) & self.uses_items.get(b, set()))
                else:
                    hit += bool(self.gate(a, b, rest, via))
            res[mode] = {"edges_explained": hit, "edges": len(supply), "share": round(hit / len(supply), 3) if supply else None}
        buys_any = {b: set(self.buys(b, supply)) for b in companies}
        pairs = sum(1 for a in companies for b in companies if a != b and self.makes(a) & buys_any[b])
        n = len(companies)
        res["gate_pass_rate_all_pairs"] = round(pairs / (n * (n - 1)), 3) if n > 1 else None
        res["companies"] = n
        res["companies_with_makes"] = sum(bool(self.makes(c)) for c in companies)
        res["companies_with_buys"] = sum(bool(buys_any[c]) for c in companies)
        res["items"] = len(self.cat)
        res["items_unmapped"] = sorted(self.kg.name(i) for i, c in self.cat.items() if c is None)
        return res

    def gate_status(self, a: str, b: str, supply: set[tuple[str, str]], primary: bool = False) -> tuple[str, set[str]]:
        """'pass' / 'fail' when both profiles are known, else 'unknown' (soft gate: unknown is not filtered out).
        On the final Phase 1 KG, where both profiles exist the gate holds for 166 of 168 existing edges
        (leave-one-out), while passing 13.7% of all company pairs."""
        mk, by = (self.makes_primary(a) if primary else self.makes(a)), set(self.buys(b, supply))
        if not mk or not by:
            return "unknown", set()
        shared = mk & by
        return ("pass" if shared else "fail"), shared

    def edge_products(self, a: str, b: str, supply: set[tuple[str, str]]) -> dict | None:
        """What A most likely sells to B along a stated A -> B supply link ("who sells what to whom").
        Categories A makes that B buys, judged WITHOUT the A -> B link itself (otherwise every link would explain
        itself through 'B buys what its supplier A makes'). Ordered by how much of A's product evidence each holds,
        so A's main product comes first. None when A's products or B's needs are unknown in the graph."""
        status, shared = self.gate_status(a, b, supply - {(a, b)})
        if status != "pass":
            return None
        w = self.makes_weighted(a)
        cats = sorted(shared, key=lambda c: (-w.get(c, 0), c))
        buys = self.buys(b, supply - {(a, b)})
        items = {c: sorted(self.kg.name(i) for i in self.makes_items.get(a, ()) if self.cat.get(i) == c)[:3] for c in cats}
        why = {}
        for c in cats:
            reasons = []
            for r in buys.get(c, []):
                kind, x = r.split(":", 1)
                reasons.append(f"uses {self.kg.name(x)}" if kind == "uses" else f"also buys from {self.kg.name(x)}")
            why[c] = sorted(set(reasons), key=lambda r: (not r.startswith("uses"), r))[:3]  # direct use first
        prim = self.makes_primary(a)
        main = [c for c in cats if c in prim] or cats[:1]   # A's main products; the rest are possible but weaker
        return {"categories": cats, "main": main, "items": items, "why": why}


def edge_flows(kg: KG, P: Products | None = None) -> dict[tuple[str, str], dict]:
    """edge_products for every stated supply link that gets a label."""
    P = P or Products(kg)
    supply = kg.supply_edges()
    out = {}
    for a, b in sorted(supply):
        r = P.edge_products(a, b, supply)
        if r:
            out[(a, b)] = r
    return out


def build(kg: KG, out_dir, strict: bool = True) -> dict:
    """Write item categories, company product profiles and the coverage report."""
    import json
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    P = Products(kg, strict)
    supply = kg.supply_edges()
    with open(out / "item_categories.jsonl", "w", encoding="utf-8") as f:
        for i, c in sorted(P.cat.items()):
            f.write(json.dumps({"item_id": i, "name": kg.name(i), "category": c}, ensure_ascii=False) + "\n")
    companies = sorted({x for e in supply for x in e} | set(P.makes_items) | set(P.uses_items))
    with open(out / "company_profiles.jsonl", "w", encoding="utf-8") as f:
        for c in companies:
            f.write(json.dumps({"company_id": c, "name": kg.name(c), "makes": sorted(P.makes(c)),
                                "buys": {k: sorted(set(v)) for k, v in sorted(P.buys(c, supply).items())}},
                               ensure_ascii=False) + "\n")
    flows = edge_flows(kg, P)
    with open(out / "edge_products.jsonl", "w", encoding="utf-8") as f:
        for (a, b), r in flows.items():
            f.write(json.dumps({"supplier_id": a, "supplier": kg.name(a), "customer_id": b, "customer": kg.name(b),
                                **r}, ensure_ascii=False) + "\n")
    cov = {"strict": strict, **P.coverage(supply),
           "supply_links_labelled": {"labelled": len(flows), "total": len(supply), "share": round(len(flows) / len(supply), 3)}}
    (out / "coverage.json").write_text(json.dumps(cov, indent=1, ensure_ascii=False), encoding="utf-8")
    return cov
