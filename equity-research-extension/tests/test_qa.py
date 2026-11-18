"""KGQA: the executor agrees with direct edge lookups, the benchmark is complete and reproducible, scoring is right."""
import os
import sys
from collections import Counter, deque
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.kg import load, qa, qa_bench  # noqa: E402

pytestmark = pytest.mark.skipif(not (load.kg_dir() / "fused_edges.jsonl").exists(), reason="Phase 1 KG not available")


@pytest.fixture(scope="module")
def env():
    kg = load.load()
    G = qa.Graph(kg)
    return kg, G, qa_bench.Bench(kg, G).build()


def test_benchmark_shape(env):
    kg, G, B = env
    assert len(B) == 120
    assert set(Counter(q["type"] for q in B).values()) == {10}
    assert len({q["question"] for q in B}) == 120
    assert Counter(q["gold"] for q in B if q["type"] == "exists") == {True: 5, False: 5}
    assert all(q["gold"] not in ([], None) for q in B if q["kind"] in ("set", "top", "hops"))


def test_reproducible(env):
    kg, G, B = env
    assert qa_bench.Bench(kg, G).build() == B


def test_gold_matches_raw_edges(env):
    kg, G, B = env
    supply = kg.supply_edges()
    sup = lambda c: {a for a, b in supply if b == c}
    cus = lambda c: {b for a, b in supply if a == c}
    for q in B:
        ids = [e["id"] for e in q["entities"]]
        if q["type"] == "suppliers":
            assert set(q["gold"]) == sup(ids[0])
        elif q["type"] == "customers":
            assert set(q["gold"]) == cus(ids[0])
        elif q["type"] == "count_customers":
            assert q["gold"] == len(cus(ids[0]))
        elif q["type"] == "exists":
            assert q["gold"] == ((ids[0], ids[1]) in supply)
        elif q["type"] == "common_suppliers":
            assert set(q["gold"]) == sup(ids[0]) & sup(ids[1])
        elif q["type"] == "hops":  # independent BFS
            dist, dq = {ids[0]: 0}, deque([ids[0]])
            while dq:
                x = dq.popleft()
                for y in cus(x):
                    if y not in dist:
                        dist[y] = dist[x] + 1
                        dq.append(y)
            assert q["gold"] == dist[ids[1]]


def test_template_planner_and_linker(env):
    kg, G, B = env
    rows = qa_bench.run(B, qa_bench.template_planner, G)
    assert qa_bench.summary(rows)["accuracy"] == 1.0
    nv = G.linker.link("NVIDIA")
    assert G.linker.link("Nvidia Corporation") == nv and G.linker.link("nvidia corp.") == nv
    assert qa.template_plan("Tell me a joke") == {"op": "unsupported"}


def test_scoring_and_bad_plans(env):
    kg, G, B = env
    q = next(q for q in B if q["kind"] == "set" and len(q["gold"]) >= 2)
    assert qa_bench.score(q, q["gold"][:1])["correct"] == 0 and 0 < qa_bench.score(q, q["gold"][:1])["f1"] < 1
    with pytest.raises(qa.PlanError):
        qa.execute({"op": "neighbors", "of": {"op": "entity", "name": "No Such Company Zzz"}, "rel": "SUPPLIES_TO"}, G)
    with pytest.raises(qa.PlanError):
        qa.execute({"op": "drop_database"}, G)
    assert qa.parse_plan('```json\n{"op":"count","of":{"op":"entity","name":"x"}}\n```')["op"] == "count"
    b = next(q for q in B if q["kind"] == "bool")
    assert qa_bench.score(b, 1)["correct"] == 0  # 1 is not True
    assert qa_bench.mcnemar([{"id": "a", "correct": 1}], [{"id": "a", "correct": 0}])["p_value"] == 1.0


def test_heldout_and_named_entities(env):
    kg, G, B = env
    H = qa_bench.Bench(kg, G, seed=11, avoid=B).build()
    assert len(H) >= 100 and not {q["question"] for q in H} & {q["question"] for q in B}
    assert all(kg.entities[e["id"]]["canonical_name"][:1].isupper() or not kg.is_company(e["id"])
               or kg.entities[e["id"]]["canonical_name"][:1].isdigit() for q in H for e in q["entities"])
    q = next(q for q in B if q["type"] == "suppliers_of_suppliers")
    assert qa_bench.score(q, q["gold"] + [q["entities"][0]["id"]])["correct"] == 1   # naming X itself is not penalised
    rows = qa_bench.run(B, qa_bench.template_planner, G)
    assert qa_bench.summary(qa_bench.rescore(rows, B), exclude={"q001"})["n"] == 119


def test_who_sells_what_to_whom(env):
    kg, G, B = env
    from src.kg import products
    P = products.Products(kg)
    supply = kg.supply_edges()
    F = products.edge_flows(kg, P)
    assert 0.3 < len(F) / len(supply) < 0.7
    for (a, b), r in list(F.items())[:40]:   # each label must hold without the link itself
        assert set(r["main"]) <= set(r["categories"]) and r["main"]
        assert set(r["categories"]) <= P.makes(a) & set(P.buys(b, supply - {(a, b)}))
    en = lambda n: {"op": "entity", "name": n}
    assert qa.execute({"op": "sells", "from": en("ASML"), "to": en("TSMC")}, G) == ["fab equipment"]
    euv = qa.execute({"op": "trades", "product": "EUV lithography systems"}, G)
    assert euv and all(kg.name(a) == "ASML" for a, _, _ in euv)
    with pytest.raises(qa.PlanError):   # no stated link: no product to report
        qa.execute({"op": "sells", "from": en("Apple"), "to": en("ASML")}, G)
    assert "trades" in qa.planner_prompt("x", 3) and "trades" not in qa.planner_prompt("x", 2)
