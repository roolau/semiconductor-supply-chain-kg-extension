from src.kg.load import KG
from src.kg.recommend import Recommender, evaluate, split


def kg():
    ents = {}
    names = ["Etch A", "Etch B", "Fab 1", "Fab 2", "Fab 3", "GPU Co", "Phone Co"]
    for i, n in enumerate(names):
        ents[f"c{i}"] = {"entity_id": f"c{i}", "canonical_name": n, "entity_type": "Company"}
    for k, n in {"i_etch": "plasma etchers", "i_fab": "foundry wafer fabrication services", "i_gpu": "GPUs",
                 "i_ph": "smartphones"}.items():
        ents[k] = {"entity_id": k, "canonical_name": n, "entity_type": "Product"}
    e = lambda s, r, o: {"subject": s, "relation": r, "object": o, "source_types": ["SEC"], "claim_ids": ["x", "y"],
                         "confidence": "HIGH"}
    edges = [e("c0", "PROVIDES", "i_etch"), e("c1", "PROVIDES", "i_etch"), e("c2", "PROVIDES", "i_fab"),
             e("c3", "PROVIDES", "i_fab"), e("c4", "PROVIDES", "i_fab"), e("c5", "PROVIDES", "i_gpu"),
             e("c6", "PROVIDES", "i_ph"),
             e("c0", "SUPPLIES_TO", "c2"), e("c0", "SUPPLIES_TO", "c3"),                         # Etch A: Fab 1, Fab 2
             e("c1", "SUPPLIES_TO", "c2"), e("c1", "SUPPLIES_TO", "c3"), e("c1", "SUPPLIES_TO", "c4"),  # Etch B also Fab 3
             e("c2", "SUPPLIES_TO", "c5"), e("c3", "SUPPLIES_TO", "c5"), e("c5", "SUPPLIES_TO", "c6")]
    return KG(ents, edges)


def test_peer_customer_is_recommended_with_evidence():
    g = kg()
    R = Recommender(g, g.supply_edges())
    recs = R.recommend("c0", 3)
    assert recs[0]["name"] == "Fab 3"                      # Etch B, its product peer, already sells to Fab 3
    ev = " ".join(recs[0]["evidence"])
    assert "Etch B" in ev and "Product fit" in ev
    assert "Phone Co" not in [r["name"] for r in recs]    # gate: Phone Co buys GPUs, not etchers


def test_supplier_direction():
    g = kg()
    R = Recommender(g, g.supply_edges(), direction="suppliers")
    assert R.recommend("c4", 1)[0]["name"] == "Etch A"     # Fab 3's peers (Fab 1, 2) buy from Etch A


def test_split_keeps_every_company_and_evaluate_runs():
    g = kg()
    S = g.supply_edges()
    train, test = split(S, 0.3, seed=1)
    assert train | test == S and not train & test
    assert {x for e in S for x in e} == {x for e in train for x in e}
    r = evaluate(g, seeds=(0, 1), frac=0.3, methods=["popularity", "hybrid"])
    assert set(r["methods"]) == {"popularity", "popularity+gate", "hybrid", "hybrid+gate"}


def test_competitors_share_products_and_customers():
    from src.kg.recommend import Competitors
    g = kg()
    C = Competitors(g)
    top = C.recommend("c0", 2)
    assert top[0][0] == "c1"                                  # Etch B: same product, same customers
    ev = top[0][2]
    assert ev["cats"] == ["fab_equipment"] and set(ev["customers"]) == {"c2", "c3"}
