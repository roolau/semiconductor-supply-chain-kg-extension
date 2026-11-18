"""Build the Phase 2 web page: the team's Phase 1 explorer plus clusters, the value-chain map and predicted relations.

    from src.kg import site; site.build("phase1_explorer.html", kg, "phase2_explorer.html")

The Phase 1 page (Team 2's "Semiconductor Supply Chain Graph") is read, never changed: its data block is kept as it
is, and this module adds a second data block (P2) plus extra markup, styles and script around it.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import viz
from .categories import CATEGORIES, LABEL, label
from .cluster import Clusters, LANES, build as cluster_build
from .load import KG
from .products import item_categories
from .recommend import Competitors, Recommender, evaluate, evaluate_competitors

TOP_K = 8


# Approximate early-November 2026 rates (US dollars per unit), only for sizing nodes on a log scale, where a few
# percent of FX movement is invisible. Sources: x-rates.com (KRW, 3 Nov), exchange-rates.org (TWD, 4 Nov),
# Dow Jones interbank table (JPY, CNY, HKD).
USD_PER = {"USD": 1.0, "KRW": 1 / 1340, "JPY": 1 / 155.8, "CNY": 1 / 6.88, "HKD": 1 / 7.82, "TWD": 1 / 32.8}
FX_NOTE = "Non-US market caps converted to USD at approximate early-November 2026 rates, for node sizes only."


def load_finance(path) -> dict | None:
    """The yfinance snapshot written by finance.write_snapshot, or None."""
    if not path or not Path(path).exists():
        return None
    snap = json.loads(Path(path).read_text(encoding="utf-8"))
    snap["fx"] = {"usd_per": USD_PER, "note": FX_NOTE}
    for d in snap["companies"].values():  # market cap in USD for node sizes (Yahoo gives it in the quote currency)
        info = d.get("info") or {}
        mc, cur = info.get("marketCap"), info.get("currency")
        if mc and cur in USD_PER:
            d["mcap_usd"] = round(mc * USD_PER[cur])
    for d in snap["companies"].values():  # weekly closes, rounded: the page needs no more
        if d.get("prices"):
            d["prices"] = [[day, round(v, 2)] for day, v in d["prices"]]
    return snap


def p2_data(kg: KG, embedding_results: dict | None = None, finance_snapshot=None) -> dict:
    m = viz.data(kg)
    recs = {}
    for direction in ("customers", "suppliers"):
        R = Recommender(kg, kg.supply_edges(), direction=direction)
        out = {}
        for a in R.nodes:
            rows = []
            for r in R.recommend(a, TOP_K):
                d = r["company_id"]
                rows.append([d, r["score"], R.evidence(a, d)])
            if rows:
                out[a] = rows
        recs[direction] = out
    C = Competitors(kg)
    recs["competitors"] = {a: rows for a in C.nodes if (rows := C.recommend(a, TOP_K))}
    ev = {d: evaluate(kg, direction=d) for d in ("customers", "suppliers")}
    C = Clusters(kg)
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        clu = cluster_build(kg, t)
    return {"map": m, "recs": recs, "lanes": LANES, "cat_label": LABEL,
            "eval": {d: {"methods": ev[d]["methods"], "test_edges": ev[d]["test_edges_per_seed"][0],
                         "seeds": len(ev[d]["seeds"]), "by_stage": ev[d]["hybrid_gate_hits10_by_stage"]} for d in ev},
            "emb": embedding_results or {}, "clusters": clu["evaluation"], "fin": load_finance(finance_snapshot),
            "eval_comp": evaluate_competitors(kg),
            "flows": C.lane_flows(), "sold": sold_data(kg),
            "itemcat": {i: c for i, c in item_categories(kg).items() if c},
            "cats": {c: label(c) for c in CATEGORIES}}


def sold_data(kg: KG) -> dict:
    """Who sells what to whom: for each labelled supply link "a|b" -> [main categories, other categories,
    {category: supplier items}, {category: why the buyer needs it}], plus coverage."""
    from .products import edge_flows
    F, total = edge_flows(kg), len(kg.supply_edges())
    links = {f"{a}|{b}": [r["main"], [c for c in r["categories"] if c not in r["main"]], r["items"], r["why"]]
             for (a, b), r in F.items()}
    return {"links": links, "labelled": len(F), "total": total}


QA_EXAMPLE_TYPES = ["suppliers", "customers", "provides", "providers_of", "partners", "suppliers_of_suppliers",
                    "downstream_2", "hops", "common_suppliers", "count_customers", "top_in_lane", "exists"]


def qa_data(qa_results=None, qa_report=None) -> dict:
    """The planner prompt (v3 = v2 + product-flow operators, split around the question) and example questions whose plans the evaluated LLM wrote
    (one correct reworded question per type, from a results_llm-*_paraphrase.jsonl file of run_qa)."""
    from .qa import planner_prompt

    head, tail = planner_prompt("<<<QUESTION>>>", 3).split("<<<QUESTION>>>")
    examples = []
    if qa_results and Path(qa_results).exists():
        rows = [json.loads(l) for l in Path(qa_results).read_text(encoding="utf-8").splitlines() if l.strip()]
        for t in QA_EXAMPLE_TYPES:
            r = next((r for r in rows if r["type"] == t and r["correct"]), None)
            if r:
                examples.append({"q": r["question"], "plan": r["plan"], "model": "gpt-oss-120b"})
    ev = None
    if qa_report and Path(qa_report).exists():
        r = json.loads(Path(qa_report).read_text(encoding="utf-8"))
        keep = ("n", "accuracy", "ci95", "mean_f1", "by_group")
        ev = {"seed": r.get("seed"), "models": r.get("models"),
              "runs": {k: {x: v[x] for x in keep} for k, v in r.items() if isinstance(v, dict) and "accuracy" in v},
              "mcnemar": r.get("mcnemar", {})}
    return {"prompt": [head, tail], "examples": examples, "eval": ev}


def build(phase1_html, kg: KG, out_path, embedding_results: dict | None = None, finance_snapshot=None,
          qa_results=None, qa_report=None) -> Path:
    html = Path(phase1_html).read_text(encoding="utf-8")
    p2 = p2_data(kg, embedding_results, finance_snapshot)
    p2["qa"] = qa_data(qa_results, qa_report)
    data = json.dumps(p2, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    tpl = Path(__file__).with_name("site_parts.html").read_text(encoding="utf-8")
    blocks = [p.split("\n", 1) for p in tpl.split("\n@@@ ")[1:]]
    keys = [k for k, _ in blocks]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:  # two blocks on the same anchor would silently replace one another
        raise ValueError(f"site_parts.html has more than one block for: {dup}")
    parts = dict(blocks)
    for key, body in parts.items():
        kind, anchor = key.split(" ", 1)
        anchor = anchor.strip().replace("\\n", "\n")
        if html.count(anchor) != 1:
            raise ValueError(f"anchor must occur exactly once in the Phase 1 page ({html.count(anchor)}x): {anchor[:60]}")
        body = body.rstrip("\n").replace("/*P2DATA*/", data)
        html = html.replace(anchor, (body + anchor) if kind == "before" else
                            (anchor + body) if kind == "after" else body, 1)
    out = Path(out_path)
    out.write_text(html, encoding="utf-8")
    return out
