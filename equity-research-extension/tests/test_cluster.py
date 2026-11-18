from src.kg.cluster import Clusters, ETO_TIER
from src.kg.eto import ETO
from src.kg.load import KG


def kg():
    ents = {}
    for i, n in enumerate(["Wafer Co", "Chip Co", "Chip Two", "Server Co", "Gas Co", "Buyer Co"]):
        ents[f"c{i}"] = {"entity_id": f"c{i}", "canonical_name": n, "entity_type": "Company"}
    items = {"i_w": "silicon wafers", "i_g": "GPUs", "i_s": "software suites", "i_srv": "servers"}
    for k, n in items.items():
        ents[k] = {"entity_id": k, "canonical_name": n, "entity_type": "Product"}
    e = lambda s, r, o, n=1: {"subject": s, "relation": r, "object": o, "source_types": ["SEC"],
                              "claim_ids": [f"x{j}" for j in range(n)], "confidence": "HIGH"}
    edges = [e("c0", "PROVIDES", "i_w"), e("c1", "PROVIDES", "i_g", 9), e("c1", "PROVIDES", "i_s", 5),
             e("c2", "PROVIDES", "i_g"), e("c3", "PROVIDES", "i_srv"),
             e("c0", "SUPPLIES_TO", "c1"), e("c0", "SUPPLIES_TO", "c2"), e("c1", "SUPPLIES_TO", "c3"),
             e("c4", "SUPPLIES_TO", "c1"),   # Gas Co: no products known, sells to a chip company
             e("c2", "SUPPLIES_TO", "c5")]   # Buyer Co: no products known, only buys chips
    return KG(ents, edges)


def test_roles_tiers_and_views():
    C = Clusters(kg())
    roles = C.role_by_products()
    assert roles == {"c0": "materials", "c1": "chips", "c2": "chips", "c3": "systems"}
    t = C.tiers()
    assert t["c4"] == {"tier": "upstream", "tier_from": "below customers"}
    assert t["c5"] == {"tier": "downstream", "tier_from": "above suppliers"}
    assert t["c1"]["tier"] == "chips" and t["c3"]["tier"] == "downstream"
    prod = C.product(0.5)
    assert prod["c1"] == prod["c2"] != prod["c0"]       # both chip makers, the wafer maker apart
    eco = C.ecosystem()
    assert set(eco) == {f"c{i}" for i in range(6)}


def test_chain_groups_win_over_software():
    C = Clusters(kg())
    assert C.role_by_products()["c1"] == "chips"         # 5 software claims do not outvote 9 chip claims


def test_eto_reference_loads():
    E = ETO()
    assert E.labels("ASML")["eto_role"] == "equipment"
    assert E.labels("Synopsys")["eto_role"] == "design_tools_ip"
    assert E.labels("Nvidia")["eto_role"] == "chip_design"
    assert E.labels("Samsung") is None                   # not gradable: only a side input in ETO
    assert set(ETO_TIER) == {"materials", "equipment", "chip_design", "design_tools_ip"}
