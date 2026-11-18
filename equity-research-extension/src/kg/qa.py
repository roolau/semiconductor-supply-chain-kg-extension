"""Module 4: question answering over the KG (KGQA).

A question is answered in three steps, and the LLM only does the middle one:
  1. the planner (rule-based templates, or an LLM) turns the question into a JSON query plan;
  2. `execute` runs the plan on the graph, linking entity names with `Linker`;
  3. the answer is written from the result alone, so nothing comes from the model's own knowledge.

The plan language is a small fixed set of operators (below, `PLAN_SPEC`), not free Cypher: the web page has no graph
database, and a fixed vocabulary can be checked before it runs. The same operators run in the page (JavaScript port).
"""
from __future__ import annotations

import difflib
import json
import re
from collections import Counter, deque

from .categories import CATEGORIES, categorize, label as cat_label
from .cluster import LANES, Clusters
from .load import KG

RELATIONS = ["SUPPLIES_TO", "PROVIDES", "USES", "PARTNERS_WITH", "SUBSIDIARY_OF", "LOCATED_IN", "OPERATES_IN"]
LANE_IDS = [k for k in LANES if k != "unknown_inputs"]
LANE_PHRASE = {"chips": "chip", "fabrication": "foundry", "assembly_test": "assembly and test",
               "systems_cloud": "systems, cloud and devices", "design_sw_ip": "design software and IP",
               "materials": "materials and chemicals", "equipment": "equipment", "components_ems": "components and EMS"}

_LANE_LIST = "; ".join(f"{k} = {LANES[k]} ({LANE_PHRASE[k]} companies)" for k in LANE_IDS)

PLAN_SPEC = f"""A plan is one JSON expression. Operators:
  {{"op":"entity","name":"<name as written in the question>"}}                       -> set with one node
  {{"op":"neighbors","of":<expr>,"rel":"<REL>","dir":"out"|"in"|"both"}}             -> set of nodes
       edges read  subject -REL-> object; "out" follows them forward, "in" backward.
       A SUPPLIES_TO B: A is B's supplier.  A PROVIDES item, A USES item.  A SUBSIDIARY_OF B: A belongs to B.
       A PARTNERS_WITH B (use "both").  A LOCATED_IN country (headquarters).  A OPERATES_IN country.
  {{"op":"within","of":<expr>,"rel":"SUPPLIES_TO","dir":"out"|"in","hops":<1-3>}}    -> nodes reachable in 1..hops steps (start excluded)
  {{"op":"intersect","args":[<expr>,<expr>]}}  {{"op":"union","args":[...]}}  {{"op":"minus","args":[<expr>,<expr>]}}
  {{"op":"lane","lane":"<LANE>"}}                                                   -> companies placed in a value-chain lane
  {{"op":"count","of":<expr>}}                                                      -> number
  {{"op":"top","of":<expr>,"by":"customers"|"suppliers","k":<int>}}                 -> list, most links first
  {{"op":"exists","from":<expr>,"rel":"<REL>","to":<expr>}}                          -> true/false (direct edge)
  {{"op":"hops","from":<expr>,"to":<expr>}}                                         -> fewest SUPPLIES_TO steps from -> to, or null
REL is one of {RELATIONS}.
LANE is one of: {_LANE_LIST}.
If the question cannot be answered with these operators, return {{"op":"unsupported"}}."""

SUFFIXES = {"inc", "corp", "corporation", "co", "ltd", "limited", "plc", "holdings", "holding", "company", "llc",
            "sa", "ag", "nv", "se", "the", "incorporated", "kk", "bv", "gmbh"}


class PlanError(ValueError):
    pass


def norm(s: str) -> str:
    s = s.lower().replace("&", " and ").replace("'s", " ")
    words = re.sub(r"[^a-z0-9぀-鿿]+", " ", s).split()
    kept = [w for w in words if w not in SUFFIXES]
    return " ".join(kept or words)


class Linker:
    """Name -> entity id: exact on normalised canonical names and aliases, then fuzzy. Ties go to the entity with
    more edges (the one a question is most likely about)."""

    def __init__(self, kg: KG):
        self.kg = kg
        deg = Counter()
        for e in kg.edges:
            deg[e["subject"]] += 1
            deg[e["object"]] += 1
        self.deg = deg
        idx: dict[str, list[str]] = {}
        for eid, e in sorted(kg.entities.items()):
            for n in [e.get("canonical_name", "")] + list(e.get("aliases") or []):
                k = norm(n)
                if k:
                    idx.setdefault(k, []).append(eid)
        self.index = {k: sorted(set(v), key=lambda x: (-deg[x], x)) for k, v in idx.items()}
        self.keys = sorted(self.index)

    def link(self, name: str) -> str | None:
        if name in self.kg.entities:
            return name
        k = norm(name)
        if k in self.index:
            return self.index[k][0]
        close = difflib.get_close_matches(k, self.keys, n=1, cutoff=0.88)
        return self.index[close[0]][0] if close else None


class Graph:
    def __init__(self, kg: KG, lanes: dict | None = None):
        self.kg = kg
        self.out: dict[str, dict[str, set]] = {r: {} for r in RELATIONS}
        self.inn: dict[str, dict[str, set]] = {r: {} for r in RELATIONS}
        for e in kg.edges:
            r = e["relation"]
            if r in self.out:
                self.out[r].setdefault(e["subject"], set()).add(e["object"])
                self.inn[r].setdefault(e["object"], set()).add(e["subject"])
        self.lanes = lanes if lanes is not None else {c: v["lane"] for c, v in Clusters(kg).lanes().items()}
        self.linker = Linker(kg)

    @property
    def flows(self) -> dict:
        """What flows along each stated supply link (products.edge_flows), computed on first use."""
        if not hasattr(self, "_flows"):
            from .products import Products, edge_flows
            self._products = Products(self.kg)
            self._flows = edge_flows(self.kg, self._products)
        return self._flows

    def product_category(self, name: str):
        """A product name or category -> (category, item id or None)."""
        n = name.strip().lower()
        for c in CATEGORIES:
            if n in (c, c.replace("_", " "), cat_label(c).lower()):
                return c, None
        eid = self.linker.link(name)
        if eid and not self.kg.is_company(eid):
            return categorize(self.kg.name(eid)), eid
        return categorize(name), None

    def nbrs(self, x: str, rel: str, d: str) -> set:
        if d == "out":
            return set(self.out[rel].get(x, ()))
        if d == "in":
            return set(self.inn[rel].get(x, ()))
        return set(self.out[rel].get(x, ())) | set(self.inn[rel].get(x, ()))

    def within(self, start: set, rel: str, d: str, hops: int) -> set:
        seen, frontier = set(start), set(start)
        for _ in range(hops):
            frontier = {y for x in frontier for y in self.nbrs(x, rel, d)} - seen
            seen |= frontier
        return seen - set(start)

    def hops(self, a: str, b: str, limit: int = 6):
        q, dist = deque([a]), {a: 0}
        while q:
            x = q.popleft()
            if x == b:
                return dist[x]
            if dist[x] < limit:
                for y in sorted(self.nbrs(x, "SUPPLIES_TO", "out")):
                    if y not in dist:
                        dist[y] = dist[x] + 1
                        q.append(y)
        return None

    def links(self, x: str, by: str) -> int:
        return len(self.nbrs(x, "SUPPLIES_TO", "out" if by == "customers" else "in"))


def execute(plan: dict, G: Graph):
    """Run a plan. Sets come back as sorted id lists, `top` as an ordered list, `sells` as category labels,
    `trades` as [supplier, customer, category] triples, plus int / bool / None."""
    v = _ev(plan, G)
    return sorted(v) if isinstance(v, set) else v


def _set(x, G, where):
    v = _ev(x, G)
    if not isinstance(v, (set, list)):
        raise PlanError(f"{where}: expected a set of nodes")
    return set(v)


def _one(x, G, where):
    s = _set(x, G, where)
    if len(s) != 1:
        raise PlanError(f"{where}: expected one entity, got {len(s)}")
    return next(iter(s))


def _ev(p, G: Graph):
    if not isinstance(p, dict) or "op" not in p:
        raise PlanError("not an operator")
    op = p["op"]
    if op == "entity":
        eid = G.linker.link(str(p.get("name", "")))
        if eid is None:
            raise PlanError(f"unknown entity: {p.get('name')!r}")
        return {eid}
    if op == "neighbors":
        rel, d = p.get("rel"), p.get("dir", "out")
        if rel not in RELATIONS or d not in ("out", "in", "both"):
            raise PlanError(f"neighbors: bad rel/dir {rel}/{d}")
        return {y for x in _set(p.get("of"), G, op) for y in G.nbrs(x, rel, d)}
    if op == "within":
        rel, d, h = p.get("rel", "SUPPLIES_TO"), p.get("dir", "out"), int(p.get("hops", 1))
        if rel not in RELATIONS or d not in ("out", "in", "both") or not 1 <= h <= 3:
            raise PlanError("within: bad arguments")
        return G.within(_set(p.get("of"), G, op), rel, d, h)
    if op in ("intersect", "union", "minus"):
        args = [_set(a, G, op) for a in p.get("args", [])]
        if len(args) < 2:
            raise PlanError(f"{op}: needs two or more args")
        out = args[0]
        for a in args[1:]:
            out = out & a if op == "intersect" else out | a if op == "union" else out - a
        return out
    if op == "lane":
        if p.get("lane") not in LANE_IDS:
            raise PlanError(f"unknown lane {p.get('lane')!r}")
        return {c for c, l in G.lanes.items() if l == p["lane"]}
    if op == "count":
        return len(_set(p.get("of"), G, op))
    if op == "top":
        by = p.get("by", "customers")
        if by not in ("customers", "suppliers"):
            raise PlanError("top: by must be customers or suppliers")
        xs = _set(p.get("of"), G, op)
        return sorted(xs, key=lambda x: (-G.links(x, by), G.kg.name(x)))[: int(p.get("k", 5))]
    if op == "exists":
        rel = p.get("rel", "SUPPLIES_TO")
        if rel not in RELATIONS:
            raise PlanError("exists: bad rel")
        a, b = _one(p.get("from"), G, op), _one(p.get("to"), G, op)
        return b in G.nbrs(a, rel, "out")
    if op == "hops":
        return G.hops(_one(p.get("from"), G, op), _one(p.get("to"), G, op))
    if op == "sells":  # product categories along the A -> B supply link, main product first
        a, b = _one(p.get("from"), G, op), _one(p.get("to"), G, op)
        if b not in G.nbrs(a, "SUPPLIES_TO", "out"):
            raise PlanError(f"no stated supply link from {G.kg.name(a)} to {G.kg.name(b)}")
        r = G.flows.get((a, b))
        return [cat_label(c) for c in r["main"]] if r else []
    if op == "trades":  # [supplier, customer, category] for every supply link that carries the product
        cat, item = G.product_category(str(p.get("product", "")))
        if not cat:
            raise PlanError(f"cannot place “{p.get('product')}” in a product category")
        sup = _set(p["from"], G, op) if p.get("from") else None
        cus = _set(p["to"], G, op) if p.get("to") else None
        G.flows  # noqa: B018 - make sure G._products exists
        out = []
        for (a, b), r in sorted(G.flows.items()):
            if cat in (r["categories"] if item else r["main"]) and (sup is None or a in sup) and (cus is None or b in cus) \
                    and (item is None or item in G._products.makes_items.get(a, ())):
                out.append([a, b, cat_label(cat)])
        return out
    if op == "unsupported":
        raise PlanError("planner: unsupported question")
    raise PlanError(f"unknown op {op!r}")


def parse_plan(text: str) -> dict:
    """Pull the JSON plan out of an LLM reply (tolerates code fences and a short preamble)."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            raise PlanError("no JSON in reply")
        return json.loads(m.group(0))


# ---------------------------------------------------------------- rule-based planner (baseline, no LLM)
_E = r"(.+?)"
TEMPLATES: list[tuple[str, callable]] = [
    (rf"^who supplies {_E}\?$", lambda a: _nb(a, "SUPPLIES_TO", "in")),
    (rf"^which companies does {_E} supply\?$", lambda a: _nb(a, "SUPPLIES_TO", "out")),
    (rf"^what does {_E} provide\?$", lambda a: _nb(a, "PROVIDES", "out")),
    (rf"^which companies provide {_E}\?$", lambda a: _nb(a, "PROVIDES", "in")),
    (rf"^who does {_E} partner with\?$", lambda a: _nb(a, "PARTNERS_WITH", "both")),
    (rf"^which companies supply the suppliers of {_E}\?$",
     lambda a: {"op": "minus", "args": [{"op": "neighbors", "of": _nb(a, "SUPPLIES_TO", "in"), "rel": "SUPPLIES_TO",
                                         "dir": "in"}, _en(a)]}),
    (rf"^which companies are within two supply steps downstream of {_E}\?$",
     lambda a: {"op": "within", "of": _en(a), "rel": "SUPPLIES_TO", "dir": "out", "hops": 2}),
    (rf"^which suppliers do {_E} and {_E} have in common\?$",
     lambda a, b: {"op": "intersect", "args": [_nb(a, "SUPPLIES_TO", "in"), _nb(b, "SUPPLIES_TO", "in")]}),
    (rf"^how many customers does {_E} have\?$", lambda a: {"op": "count", "of": _nb(a, "SUPPLIES_TO", "out")}),
    (rf"^does {_E} supply {_E}\?$", lambda a, b: {"op": "exists", "from": _en(a), "rel": "SUPPLIES_TO", "to": _en(b)}),
    (r"^which (.+?) company has the most (customers|suppliers)\?$",
     lambda lane, by: {"op": "top", "of": {"op": "lane", "lane": _lane(lane)}, "by": by, "k": 1}),
    (rf"^how many supply steps separate {_E} from {_E}\?$", lambda a, b: {"op": "hops", "from": _en(a), "to": _en(b)}),
]
# product-flow phrasings: not part of the 12 benchmark templates, used by the page parser
PRODUCT_TEMPLATES = [
    (rf"^what does {_E} (?:sell|supply|provide) to {_E}\?$", lambda a, b: {"op": "sells", "from": _en(a), "to": _en(b)}),
    (rf"^who sells {_E} to whom\?$", lambda x: {"op": "trades", "product": x}),
]


def _en(a):
    return {"op": "entity", "name": a}


def _nb(a, rel, d):
    return {"op": "neighbors", "of": _en(a), "rel": rel, "dir": d}


def _lane(label):
    for k, v in LANES.items():
        if label.lower() in (k, v.lower(), LANE_PHRASE.get(k, "").lower()):
            return k
    return label


def template_plan(question: str) -> dict:
    q = " ".join(question.strip().split())
    for pat, build in TEMPLATES:
        m = re.match(pat, q, re.I)
        if m:
            return build(*m.groups())
    return {"op": "unsupported"}


# ---------------------------------------------------------------- LLM planner prompt (model-agnostic)
PROMPT_RULES_V2 = """Rules:
- Products, inputs and technologies are entities too. "Which firms make X?" is neighbors(entity X, PROVIDES, in),
  with X copied as written, even when X sounds generic. Do not answer it with a lane.
- A company's customers are the companies it supplies: neighbors(X, SUPPLIES_TO, out). Its suppliers:
  neighbors(X, SUPPLIES_TO, in). "Provides / sells / delivers to <companies>" is SUPPLIES_TO; PROVIDES only links
  a company to a product.
- "Tier-2 suppliers" or "suppliers of the suppliers" of X: neighbors of neighbors, both SUPPLIES_TO "in".
- Use {"op":"unsupported"} only when no operator fits, never because a name looks unfamiliar."""

EXAMPLES_V2 = (
    'Q: How many customers does Intel have?\nA: {"op":"count","of":{"op":"neighbors","of":{"op":"entity","name":"Intel"},'
    '"rel":"SUPPLIES_TO","dir":"out"}}\n'
    'Q: Which firms make EUV lithography systems?\nA: {"op":"neighbors","of":{"op":"entity","name":"EUV lithography systems"},'
    '"rel":"PROVIDES","dir":"in"}\n'
)


PRODUCT_OPS = """Product flows (who sells what to whom; inferred from what the supplier makes and the buyer needs):
  {"op":"sells","from":<expr>,"to":<expr>}                    -> A's main product categories that B buys, along a stated A -> B supply link
  {"op":"trades","product":"<product or category>","from":<expr, optional>,"to":<expr, optional>}
                                                             -> [supplier, customer, category] links carrying it
Examples: "What does TSMC sell to NVIDIA?" -> {"op":"sells","from":{"op":"entity","name":"TSMC"},"to":{"op":"entity","name":"NVIDIA"}}
"Who sells EUV lithography systems to whom?" -> {"op":"trades","product":"EUV lithography systems"}"""


def planner_prompt(question: str, version: int = 1) -> str:
    """version 1: the prompt of the first evaluation run (kept unchanged so its cached replies stay valid).
    version 2: adds the rules and two examples written after the error analysis of run 1 (seed 7 benchmark)."""
    if version == 3:  # the web page: v2 plus the product-flow operators (not part of the v1/v2 evaluation)
        base = planner_prompt(question, 2)
        return base.replace("\n\nRules:\n", "\n" + PRODUCT_OPS + "\n\nRules:\n", 1)
    if version == 2:
        base = planner_prompt(question, 1)
        head, tail = base.split("Copy entity names exactly", 1)
        return head.replace("\n\nExamples:\n", "\n\n" + PROMPT_RULES_V2 + "\n\nExamples:\n", 1).replace(
            "Q: What does TSMC use?", EXAMPLES_V2 + "Q: What does TSMC use?", 1) + "Copy entity names exactly" + tail
    return (
        "You translate questions about a semiconductor supply-chain knowledge graph into a query plan.\n"
        f"{PLAN_SPEC}\n\n"
        "Examples:\n"
        'Q: Who supplies Intel?\nA: {"op":"neighbors","of":{"op":"entity","name":"Intel"},"rel":"SUPPLIES_TO","dir":"in"}\n'
        'Q: Is Applied Materials a supplier to Samsung?\nA: {"op":"exists","from":{"op":"entity","name":"Applied Materials"},'
        '"rel":"SUPPLIES_TO","to":{"op":"entity","name":"Samsung"}}\n'
        'Q: What does TSMC use?\nA: {"op":"neighbors","of":{"op":"entity","name":"TSMC"},"rel":"USES","dir":"out"}\n\n'
        "Copy entity names exactly as they appear in the question. Reply with the JSON plan only.\n"
        f"Q: {question}\nA:"
    )


def answer_text(value, G: Graph, max_items: int = 15) -> str:
    """Plain answer from the result only (used by the page and for spot checks)."""
    if value is None:
        return "Not connected in the graph."
    if isinstance(value, bool):
        return "Yes, according to the graph." if value else "No such link in the graph."
    if isinstance(value, int):
        return str(value)
    names = [G.kg.name(x) for x in value]
    if not names:
        return "Nothing in the graph."
    more = f" and {len(names) - max_items} more" if len(names) > max_items else ""
    return ", ".join(names[:max_items]) + more
