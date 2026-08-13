from __future__ import annotations

import re


_CLASS_SYNONYMS: dict[str, list[str]] = {
    "low": [
        "low", "budget", "cheap", "inexpensive", "affordable",
        "entry-level", "entry level", "low-end", "low end", "value-priced",
    ],
    "mid": [
        "mid", "mid-range", "mid range", "moderate", "moderately priced",
        "mid-tier", "mid tier", "average-priced", "middle of the road",
    ],
    "high": [
        "high", "premium", "expensive", "high-end", "high end",
        "top-tier", "top tier", "luxury", "flagship", "pricey",
    ],
}


def _find_matches(text_lower: str, terms: list[str]) -> list[str]:
    matches = []
    for term in terms:
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, text_lower):
            matches.append(term)
    return matches


def check_class_alignment(predicted_class: str, text: str) -> dict:

    key = predicted_class.strip().lower()
    text_lower = text.lower()
    reasons: list[str] = []

    own_terms = _CLASS_SYNONYMS.get(key, [key])
    own_matches = _find_matches(text_lower, own_terms)

    contradictions: dict[str, list[str]] = {}
    for other_class, terms in _CLASS_SYNONYMS.items():
        if other_class == key:
            continue
        hits = _find_matches(text_lower, terms)
        if hits:
            contradictions[other_class] = hits

    if not own_matches:
        reasons.append(
            f"Text never uses '{predicted_class}' or a recognized "
            f"synonym for the predicted class."
        )
    for other_class, hits in contradictions.items():
        reasons.append(
            f"Text uses language associated with '{other_class}' "
            f"({', '.join(hits)}) despite predicted class being "
            f"'{predicted_class}'."
        )

    if contradictions:
        status = "failed"
    elif own_matches:
        status = "passed"
    else:
        status = "uncertain"

    return {
        "is_aligned": status == "passed",
        "status": status,
        "reasons": reasons,
    }