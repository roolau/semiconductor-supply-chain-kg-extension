from src.kg.categories import categorize
from src.kg.load import KG
from src.kg.products import Products


def test_categorize_specific_before_general():
    assert categorize("memory test systems") == "test_equipment"
    assert categorize("photoresist") == "process_chemicals"
    assert categorize("EUV lithography systems") == "fab_equipment"
    assert categorize("HBM") == "memory_chips"
    assert categorize("GDDR SDRAM") == "memory_chips"
    assert categorize("power integrated circuits") == "analog_power_chips"
    assert categorize("multi-chip packages") == "packaging_osat"
    assert categorize("deposition materials") == "process_chemicals"
    assert categorize("services and support") is None  # too generic: never guessed


def kg():
    ents = {f"c{i}": {"entity_id": f"c{i}", "canonical_name": f"C{i}", "entity_type": "Company"} for i in range(1, 6)}
    ents.update({"i1": {"entity_id": "i1", "canonical_name": "EUV photoresist", "entity_type": "Input"},
                 "i2": {"entity_id": "i2", "canonical_name": "photoresist", "entity_type": "Input"},
                 "i3": {"entity_id": "i3", "canonical_name": "GPUs", "entity_type": "Product"}})
    e = lambda s, r, o, src=("SEC",): {"subject": s, "relation": r, "object": o, "source_types": list(src),
                                       "claim_ids": ["x"], "confidence": "MEDIUM"}
    edges = [e("c1", "PROVIDES", "i1"), e("c2", "USES", "i2"),      # step 2: item differs, category matches
             e("c3", "PROVIDES", "i2"), e("c3", "SUPPLIES_TO", "c4"),  # step 1: c4 buys photoresist via supplier c3
             e("c5", "PROVIDES", "i3"),
             e("c5", "PROVIDES", "i1", ("WIKIPEDIA",))]                  # weak: 1 Wikipedia claim, MEDIUM
    return KG(ents, edges)


def test_two_steps_and_soft_gate():
    g = kg()
    P = Products(g)
    S = g.supply_edges()
    assert P.gate("c1", "c2", S) == {"process_chemicals"}           # category match, not item match
    assert P.gate_status("c1", "c4", S) == ("pass", {"process_chemicals"})  # buys inferred from its supplier
    assert P.gate_status("c5", "c4", S)[0] == "fail"                 # GPU maker vs photoresist buyer
    assert P.gate_status("c1", "c5", S)[0] == "unknown"              # c5 buys nothing known: not filtered
    assert P.gate_status("c1", "c4", S - {("c3", "c4")})[0] == "unknown"  # leave-one-out removes the evidence


def test_strict_drops_weak_item_edges():
    g = kg()
    assert "process_chemicals" not in Products(g).makes("c5")
    assert "process_chemicals" in Products(g, strict=False).makes("c5")
