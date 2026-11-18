"""Company-name normalisation used to match names across the KG and the ETO reference data."""
from __future__ import annotations

import re
import unicodedata

LEGAL_SUFFIX = re.compile(
    r"\b(incorporated|inc|corporation|corp|company|co|limited|ltd|llc|plc|n ?v|s ?a|ag|se|gmbh|kg|bv|b ?v|"
    r"holdings?|group|the|sa de cv|co ltd|kk|pte|pty|oyj|ab|asa|spa|s ?p ?a)\b")

DE_SUFFIX = re.compile(r"\s*/\s*[a-z]{2}\s*$")  # SEC style "APPLIED MATERIALS INC /DE"


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    for a, b in [("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"), ("–", "-"), ("—", "-"),
                 ("®", ""), ("™", ""), ("©", "")]:
        s = s.replace(a, b)
    return s


def norm_name(s: str) -> str:
    s = fold(s).lower()
    s = DE_SUFFIX.sub("", s)
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\s+", " ", LEGAL_SUFFIX.sub(" ", s)).strip()
    return s
