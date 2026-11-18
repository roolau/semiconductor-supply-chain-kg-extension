"""Read the final Phase 1 KG (read-only).

Final = the reviewed copy of 13 Oct 2026 (`semiconductor_kg/data/kg/`, summary.json "reviewed_copy": 2,988 edges,
1,598 active entities). Files used: fused_edges.jsonl, canonical_entities.jsonl.
Location: env P1_KG_DIR, else env KG_ROOT/data/kg, else the repository's data/ folder (this extension sits in the repo).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


def kg_dir() -> Path:
    if os.environ.get("P1_KG_DIR"):
        return Path(os.environ["P1_KG_DIR"])
    if os.environ.get("KG_ROOT"):
        return Path(os.environ["KG_ROOT"]) / "data" / "kg"
    return Path(__file__).resolve().parents[3] / "data"   # repo/data, next to equity-research-extension/


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


@dataclass
class KG:
    entities: dict[str, dict]
    edges: list[dict]
    by_relation: dict[str, list[dict]] = field(default_factory=dict)

    def __post_init__(self):
        for e in self.edges:
            self.by_relation.setdefault(e["relation"], []).append(e)

    def name(self, eid: str) -> str:
        return (self.entities.get(eid) or {}).get("canonical_name") or eid

    def is_company(self, eid: str) -> bool:
        return (self.entities.get(eid) or {}).get("entity_type") == "Company"

    def supply_edges(self) -> set[tuple[str, str]]:
        """(supplier, customer) pairs from SUPPLIES_TO, both ends companies, no self loops."""
        return {(e["subject"], e["object"]) for e in self.by_relation.get("SUPPLIES_TO", [])
                if self.is_company(e["subject"]) and self.is_company(e["object"]) and e["subject"] != e["object"]}


def load(path: str | Path | None = None) -> KG:
    d = Path(path) if path else kg_dir()
    ents = {r["entity_id"]: r for r in _jsonl(d / "canonical_entities.jsonl")}
    return KG(ents, _jsonl(d / "fused_edges.jsonl"))
