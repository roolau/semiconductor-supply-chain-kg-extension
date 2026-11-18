"""KGQA benchmark: 120 template questions with gold plans and gold answers computed from the KG (no LLM), an optional
LLM paraphrase of each question, and scoring.

12 question types x 10 questions, seeded. About a third of entity mentions use an alias instead of the canonical name,
so entity linking is tested too. Gold answers come from running the gold plan; `tests/test_qa.py` checks the executor
against direct edge lookups so the gold is not just the executor agreeing with itself.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

from .load import KG
from .qa import LANE_IDS, LANE_PHRASE, Graph, PlanError, execute, norm, template_plan

TYPES = {  # type: (group, answer kind)
    "suppliers": ("one-hop", "set"), "customers": ("one-hop", "set"), "provides": ("one-hop", "set"),
    "providers_of": ("one-hop", "set"), "partners": ("one-hop", "set"),
    "suppliers_of_suppliers": ("multi-hop", "set"), "downstream_2": ("multi-hop", "set"), "hops": ("multi-hop", "hops"),
    "common_suppliers": ("set", "set"), "count_customers": ("aggregate", "count"), "top_in_lane": ("aggregate", "top"),
    "exists": ("boolean", "bool"),
}
PER_TYPE = 10

# Questions found invalid after a run, by benchmark seed. They stay in the file (so ids and the cache stay stable) but
# are left out of the summary. Seed 7 was generated before the generator learned to skip such entities.
KNOWN_DEFECTS = {7: {"q057": "the entity is a phrase ('most manufacturers of computer systems') that the Phase 1 "
                             "extraction mistook for a company"}}


class Bench:
    def __init__(self, kg: KG, G: Graph | None = None, seed: int = 7, alias_share: float = 1 / 3,
                 avoid: list[dict] | None = None):
        """avoid: an earlier benchmark whose questions must not reappear (same type and same entities), so a
        held-out benchmark shares nothing with the development one."""
        self.kg, self.G = kg, G or Graph(kg)
        self.avoid = {(q["type"], tuple(e["id"] for e in q["entities"]), json.dumps(q["plan"].get("of", {}).get("lane")))
                      for q in avoid or []}
        self.rng = random.Random(seed)
        self.alias_share = alias_share
        self.companies = sorted(c for c in self.G.lanes)  # companies in the supply graph

    # surface form: canonical name, or (sometimes) an alias that still links back to the same entity
    def surface(self, eid: str) -> str | None:
        e = self.kg.entities[eid]
        name = e["canonical_name"]
        company = self.kg.is_company(eid)
        if self.G.linker.link(name) != eid:
            return None  # ambiguous name: skip this entity
        if company and not name[:1].isupper() and not name[:1].isdigit():
            return None  # a description extracted as a company ("most manufacturers of ..."), not a name
        aliases = sorted({a for a in e.get("aliases") or [] if company and norm(a) != norm(name) and 2 < len(a) <= 40
                          and a.isascii() and self.G.linker.link(a) == eid})
        if aliases and self.rng.random() < self.alias_share:
            return self.rng.choice(aliases)
        return name

    def _pick(self, pool, k=PER_TYPE):
        pool = sorted(pool, key=str)
        self.rng.shuffle(pool)
        return pool

    def build(self) -> list[dict]:
        G, out = self.G, []
        nb = G.nbrs
        sup = lambda c: nb(c, "SUPPLIES_TO", "in")
        cus = lambda c: nb(c, "SUPPLIES_TO", "out")

        def add(qtype, question, plan, ents, want=PER_TYPE):
            if sum(1 for q in out if q["type"] == qtype) >= want:
                return False
            if any(s is None for s, _ in ents):
                return False
            if (qtype, tuple(i for _, i in ents), json.dumps(plan.get("of", {}).get("lane"))) in self.avoid:
                return False
            try:
                gold = execute(plan, G)
            except PlanError:
                return False
            out.append({"type": qtype, "group": TYPES[qtype][0], "kind": TYPES[qtype][1], "question": question,
                        "entities": [{"surface": s, "id": i} for s, i in ents], "plan": plan, "gold": gold})
            return True

        en = lambda s: {"op": "entity", "name": s}
        nbp = lambda s, rel, d: {"op": "neighbors", "of": en(s), "rel": rel, "dir": d}

        for c in self._pick([c for c in self.companies if 2 <= len(sup(c)) <= 15]):
            s = self.surface(c); add("suppliers", f"Who supplies {s}?", nbp(s, "SUPPLIES_TO", "in"), [(s, c)])
        for c in self._pick([c for c in self.companies if 2 <= len(cus(c)) <= 15]):
            s = self.surface(c); add("customers", f"Which companies does {s} supply?", nbp(s, "SUPPLIES_TO", "out"), [(s, c)])
        for c in self._pick([c for c in self.companies if 2 <= len(nb(c, "PROVIDES", "out")) <= 12]):
            s = self.surface(c); add("provides", f"What does {s} provide?", nbp(s, "PROVIDES", "out"), [(s, c)])
        items = {x for c in self.companies for x in nb(c, "PROVIDES", "out")}
        for i in self._pick([i for i in items if 2 <= len(nb(i, "PROVIDES", "in")) <= 10]):
            s = self.surface(i); add("providers_of", f"Which companies provide {s}?", nbp(s, "PROVIDES", "in"), [(s, i)])
        for c in self._pick([c for c in self.companies if 2 <= len(nb(c, "PARTNERS_WITH", "both")) <= 12]):
            s = self.surface(c); add("partners", f"Who does {s} partner with?", nbp(s, "PARTNERS_WITH", "both"), [(s, c)])
        for c in self._pick([c for c in self.companies if 2 <= len({y for x in sup(c) for y in sup(x)} - {c}) <= 20]):
            s = self.surface(c)
            add("suppliers_of_suppliers", f"Which companies supply the suppliers of {s}?",
                {"op": "minus", "args": [{"op": "neighbors", "of": nbp(s, "SUPPLIES_TO", "in"), "rel": "SUPPLIES_TO",
                                          "dir": "in"}, en(s)]}, [(s, c)])
        for c in self._pick([c for c in self.companies
                             if 3 <= len(G.within({c}, "SUPPLIES_TO", "out", 2)) <= 25
                             and len(G.within({c}, "SUPPLIES_TO", "out", 2)) > len(cus(c))]):
            s = self.surface(c)
            add("downstream_2", f"Which companies are within two supply steps downstream of {s}?",
                {"op": "within", "of": en(s), "rel": "SUPPLIES_TO", "dir": "out", "hops": 2}, [(s, c)])
        # hops: 5 pairs at distance 2, 3 at 3, 2 at 4
        pairs = {2: [], 3: [], 4: []}
        for a in self.companies:
            for b in self.companies:
                if a != b:
                    h = G.hops(a, b, limit=4)
                    if h in pairs:
                        pairs[h].append((a, b))
        for h, want in ((2, 5), (3, 8), (4, 10)):
            for a, b in self._pick(pairs[h]):
                if sum(1 for q in out if q["type"] == "hops") >= want:
                    break
                sa, sb = self.surface(a), self.surface(b)
                add("hops", f"How many supply steps separate {sa} from {sb}?",
                    {"op": "hops", "from": en(sa), "to": en(sb)}, [(sa, a), (sb, b)])
        cand = {(a, b) for s0 in self.companies for a in cus(s0) for b in cus(s0) if a < b}
        for a, b in self._pick([p for p in cand if 1 <= len(sup(p[0]) & sup(p[1])) <= 10]):
            sa, sb = self.surface(a), self.surface(b)
            add("common_suppliers", f"Which suppliers do {sa} and {sb} have in common?",
                {"op": "intersect", "args": [nbp(sa, "SUPPLIES_TO", "in"), nbp(sb, "SUPPLIES_TO", "in")]},
                [(sa, a), (sb, b)])
        seen_counts = Counter()
        for c in self._pick([c for c in self.companies if cus(c)]):
            n = len(cus(c))
            if seen_counts[min(n, 6)] >= 2:  # spread the answers instead of ten companies with one customer
                continue
            s = self.surface(c)
            if add("count_customers", f"How many customers does {s} have?",
                   {"op": "count", "of": nbp(s, "SUPPLIES_TO", "out")}, [(s, c)]):
                seen_counts[min(n, 6)] += 1
        combos = []
        for lane in LANE_IDS:
            members = [c for c, l in G.lanes.items() if l == lane]
            if len(members) < 3:
                continue
            for by in ("customers", "suppliers"):
                best = max(G.links(c, by) for c in members)
                if best > 0:
                    combos.append((lane, by, sorted(c for c in members if G.links(c, by) == best)))
        combos.sort(key=lambda x: (len(x[2]) > 1, x[0], x[1]))  # unique winners first
        for lane, by, tied in combos:
            if add("top_in_lane", f"Which {LANE_PHRASE[lane]} company has the most {by}?",
                   {"op": "top", "of": {"op": "lane", "lane": lane}, "by": by, "k": 1}, []):
                out[-1]["gold"] = tied
        true_edges = self._pick(sorted(self.kg.supply_edges()))
        for a, b in true_edges[:40]:
            if sum(1 for q in out if q["type"] == "exists" and q["gold"]) >= PER_TYPE // 2:
                break
            sa, sb = self.surface(a), self.surface(b)
            add("exists", f"Does {sa} supply {sb}?", {"op": "exists", "from": en(sa), "rel": "SUPPLIES_TO", "to": en(sb)},
                [(sa, a), (sb, b)])
        # hard negatives: b buys from a company that shares a customer with a, but not from a
        negs = sorted({(a, b) for a in self.companies for p in self.companies if p != a and cus(a) & cus(p)
                       for b in cus(p) if b not in cus(a) and b != a})
        for a, b in self._pick(negs):
            sa, sb = self.surface(a), self.surface(b)
            add("exists", f"Does {sa} supply {sb}?", {"op": "exists", "from": en(sa), "rel": "SUPPLIES_TO", "to": en(sb)},
                [(sa, a), (sb, b)])
        for i, q in enumerate(sorted(out, key=lambda q: list(TYPES).index(q["type"]))):
            q["id"] = f"q{i + 1:03d}"
        out.sort(key=lambda q: q["id"])
        for q in out:
            q["gold_names"] = _names(q["gold"], self.kg)
        return [{"id": q["id"], **{k: v for k, v in q.items() if k != "id"}} for q in out]


def _names(v, kg):
    return [kg.name(x) for x in v] if isinstance(v, list) else v


# ---------------------------------------------------------------- scoring
def score(q: dict, value) -> dict:
    """correct (0/1) plus F1 for set answers. `value` is what execute() returned, or an exception."""
    if isinstance(value, Exception):
        return {"correct": 0, "f1": 0.0, "error": str(value)}
    kind, gold = q["kind"], q["gold"]
    if kind == "set":
        if not isinstance(value, list):
            return {"correct": 0, "f1": 0.0, "error": "wrong answer type"}
        # the entities named in the question never count either way: "who supplies the suppliers of X" may or may
        # not list X itself when X sits in a supply loop, and both readings are fair
        named = {e["id"] for e in q["entities"]}
        p, g = set(value) - named, set(gold) - named
        tp = len(p & g)
        f1 = 2 * tp / (len(p) + len(g)) if p or g else 1.0
        return {"correct": int(p == g), "f1": round(f1, 4)}
    if kind == "top":
        ok = isinstance(value, list) and bool(value) and value[0] in gold
        return {"correct": int(ok), "f1": float(ok)}
    ok = (type(value) is type(gold) or (gold is None and value is None)) and value == gold
    if kind == "count" and isinstance(value, list):
        ok = False
    return {"correct": int(ok), "f1": float(ok)}


def run(bench: list[dict], planner, G: Graph, field: str = "question") -> list[dict]:
    """planner(question) -> plan dict. Returns one row per question with the plan, answer and score."""
    rows = []
    for q in bench:
        plan = None
        try:
            plan = planner(q[field])
            value = execute(plan, G)
        except Exception as e:  # noqa: BLE001 - an unusable plan is a wrong answer, not a crash
            if type(e).__name__ == "LLMUnavailable":
                raise  # quota or API failure: stop; cached replies make the rerun resume here
            plan, value = locals().get("plan"), e
        rows.append({"id": q["id"], "type": q["type"], "group": q["group"], "question": q[field], "plan": plan,
                     "answer": None if isinstance(value, Exception) else value, **score(q, value)})
    return rows


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def mcnemar(a: list[dict], b: list[dict], exclude=()) -> dict:
    """Exact McNemar test on paired correctness (same questions, two systems)."""
    ca = {r["id"]: r["correct"] for r in a if r["id"] not in set(exclude)}
    cb = {r["id"]: r["correct"] for r in b if r["id"] not in set(exclude)}
    b10 = sum(1 for i in ca if ca[i] and not cb[i])
    b01 = sum(1 for i in ca if not ca[i] and cb[i])
    n = b10 + b01
    k = min(b10, b01)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"only_first_correct": b10, "only_second_correct": b01, "p_value": round(p, 5)}


def rescore(rows: list[dict], bench: list[dict]) -> list[dict]:
    """Score saved result rows again (after a scoring change) without calling any planner."""
    gold = {q["id"]: q for q in bench}
    out = []
    for r in rows:
        q = gold[r["id"]]
        value = PlanError(r["error"]) if r.get("error") else r["answer"]
        out.append({**{k: v for k, v in r.items() if k not in ("correct", "f1", "error")}, **score(q, value)})
    return out


def summary(rows: list[dict], exclude=()) -> dict:
    rows = [r for r in rows if r["id"] not in set(exclude)]
    n, k = len(rows), sum(r["correct"] for r in rows)
    out = {"n": n, "accuracy": round(k / n, 3) if n else 0, "ci95": wilson(k, n),
           "mean_f1": round(sum(r["f1"] for r in rows) / n, 3) if n else 0,
           "errors": sum(1 for r in rows if r.get("error"))}
    for key in ("group", "type"):
        g = {}
        for r in rows:
            g.setdefault(r[key], []).append(r["correct"])
        out[f"by_{key}"] = {x: round(sum(v) / len(v), 3) for x, v in sorted(g.items())}
    return out


# ---------------------------------------------------------------- paraphrases (LLM, run in Colab)
def paraphrase_prompt(batch: list[dict]) -> str:
    lines = "\n".join(f'{q["id"]}: {q["question"]}' for q in batch)
    return (
        "Rewrite each question the way a business analyst might ask it in a chat box: vary the wording and sentence "
        "structure, but keep the exact meaning and keep every company or product name exactly as written (same "
        "spelling). Do not add or drop conditions. Return a JSON object mapping each id to its rewritten question.\n\n"
        + lines
    )


def check_paraphrase(q: dict, text: str) -> bool:
    """Accept a paraphrase only if every entity mention survived verbatim (case-insensitive)."""
    t = text.lower()
    return bool(text.strip()) and all(e["surface"].lower() in t for e in q["entities"])


def build(kg: KG, out_path, seed: int = 7, avoid: list[dict] | None = None) -> list[dict]:
    bench = Bench(kg, seed=seed, avoid=avoid).build()
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in bench) + "\n")
    return bench


def load_bench(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def paraphrased(bench: list[dict], old: list[dict]) -> int:
    """Copy paraphrases from an earlier file of the same benchmark (matched on question text). Returns how many."""
    prev = {q["question"]: q for q in old if q.get("paraphrase")}
    n = 0
    for q in bench:
        if q["question"] in prev:
            q["paraphrase"], q["paraphrase_ok"] = prev[q["question"]]["paraphrase"], prev[q["question"]]["paraphrase_ok"]
            n += 1
    return n


def template_planner(question: str) -> dict:
    return template_plan(question)
