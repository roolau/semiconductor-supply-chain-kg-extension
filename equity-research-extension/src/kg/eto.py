"""CSET ETO Supply Chain Explorer as the reference for supply-chain roles (data/reference/eto/, CC BY-NC 4.0).

ETO lists which organisation provides which input (design resource, tool, material). From that, each matched KG company
gets two reference labels:
  eto_role   chip_design | design_tools_ip | equipment | materials      (from the input types it provides)
  eto_stage  Design | Fabrication | ATP                                  (the stage those inputs feed, via sequence.csv)
Limitation, stated in the report: ETO models inputs, not chip making itself. Fabs and OSATs appear only through
captive inputs (TSMC, Samsung, SMIC, GlobalFoundries: photomasks). Photomask-only providers are therefore dropped from
the reference, and "foundry" / "OSAT" roles cannot be graded against ETO.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from .names import norm_name

ETO_DIR = Path(__file__).resolve().parents[2] / "data" / "reference" / "eto"
STAGE_NAME = {"S1": "Design", "S2": "Fabrication", "S3": "ATP"}
CAPTIVE_ONLY = {"N33"}  # photomasks: fabs' captive mask shops, says nothing about their role
DESIGN_TOOLS = {"N84", "N85", "N7"}  # EDA software, core IP, "EDA and core IP"
# ETO lists these only through one side input, which would grade them by a role they do not have
NOT_GRADABLE = {"Samsung": "IDM; ETO lists only package substrates", "Amkor": "OSAT; ETO lists only ceramic packages"}


def _rows(name: str, d: Path) -> list[dict]:
    with open(d / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class ETO:
    def __init__(self, d: Path | str = ETO_DIR):
        d = Path(d)
        self.inputs = {r["input_id"]: r for r in _rows("inputs.csv", d)}
        self.parent = defaultdict(list)
        for r in _rows("sequence.csv", d):
            for k in ("goes_into_id", "is_type_of_id"):
                if r.get(k):
                    self.parent[r["input_id"]].append(r[k])
        orgs = {r["provider_id"]: r["provider_name"] for r in _rows("providers.csv", d)
                if r["provider_type"] == "organization"}
        self.provides: dict[str, set[str]] = defaultdict(set)
        for r in _rows("provision.csv", d):
            if r["provider_id"] in orgs:
                self.provides[orgs[r["provider_id"]]].add(r["provided_id"])
        self._stage = lru_cache(maxsize=None)(self._stage_of)

    def _stage_of(self, iid: str, depth: int = 0) -> str | None:
        sid = self.inputs.get(iid, {}).get("stage_id")
        if sid:
            return STAGE_NAME.get(sid)
        if depth > 10:
            return None
        for p in self.parent.get(iid, []):
            s = self._stage_of(p, depth + 1)
            if s:
                return s
        return None

    def role_of_input(self, iid: str) -> str | None:
        t = self.inputs.get(iid, {}).get("type")
        if iid in DESIGN_TOOLS:
            return "design_tools_ip"
        return {"design_resource": "chip_design", "tool_resource": "equipment", "material_resource": "materials"}.get(t)

    def labels(self, org: str) -> dict | None:
        if org in NOT_GRADABLE:
            return None
        ids = self.provides.get(org, set()) - CAPTIVE_ONLY
        if not ids:
            return None
        ids = sorted(ids)  # fixed order: most_common breaks ties by first seen, so runs agree
        roles = Counter(r for r in (self.role_of_input(i) for i in ids) if r)
        stages = Counter(s for s in (self._stage(i) for i in ids) if s)
        if not roles:
            return None
        return {"eto_org": org, "eto_role": roles.most_common(1)[0][0], "eto_roles": dict(roles),
                "eto_stage": stages.most_common(1)[0][0] if stages else None, "eto_stages": dict(stages),
                "n_inputs": len(ids)}

    def match(self, kg) -> dict[str, dict]:
        """KG company_id -> ETO labels, matched on normalized canonical name or alias (exact)."""
        index = {norm_name(o): o for o in self.provides}
        out = {}
        for eid, ent in kg.entities.items():
            if ent.get("entity_type") != "Company":
                continue
            for n in [ent.get("canonical_name")] + list(ent.get("aliases") or []):
                o = index.get(norm_name(n or ""))
                if o:
                    lab = self.labels(o)
                    if lab:
                        out[eid] = lab
                    break
        return out
