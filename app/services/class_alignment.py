from __future__ import annotations

import re

# predicted_class comes from a 3-bucket quantile split -- "low" / "mid" / "high"
# (see app/data/preprocessing/create_labels.py). A report can talk about the
# right price tier without ever using that literal word ("premium" for
# "high"), and a report that uses language for a *different* tier is much
# stronger evidence of misalignment than the predicted word simply being
# absent. Both cases were indistinguishable under a single substring check.
#
# This used to be duplicated: GuardrailAgent had this exact logic, and
# ReportService independently had a cruder version of the same idea
# (`predicted_class.lower() not in report.lower()` -- no word boundaries,
# no synonyms, no contradiction detection) to decide whether to discard the
# LLM's report and substitute a canned fallback. Two copies of "does this
# text align with the predicted class," one of them much weaker, checking
# the same thing for related but different purposes. Now there's one.
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
    """Checks whether `text` aligns with `predicted_class`.

    Returns {"is_aligned": bool, "status": "passed"|"uncertain"|"failed",
    "reasons": list[str]}.

    - "passed": text uses the predicted class's own term or a synonym.
    - "failed": text uses language for a *different* class -- an explicit
      contradiction, the strongest signal of misalignment.
    - "uncertain": text uses no class language at all either way. Weaker
      evidence than a contradiction -- callers should decide for
      themselves whether silence is acceptable for their use case.
    """
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