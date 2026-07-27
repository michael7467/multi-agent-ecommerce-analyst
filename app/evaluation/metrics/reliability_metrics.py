from __future__ import annotations


def outcome_rate(results: list[dict], outcome_key: str, target_value) -> float:
    """Fraction of results where results[i][outcome_key] == target_value.

    0.0 for an empty results list, matching what both existing call sites
    already did rather than raising a divide-by-zero -- an eval run over
    zero cases has no rate to report, not an undefined one worth crashing
    over.
    """
    n = len(results)
    if n == 0:
        return 0.0
    matches = sum(1 for r in results if r.get(outcome_key) == target_value)
    return matches / n


def outcome_rates(results: list[dict], outcome_key: str, possible_values: list[str]) -> dict[str, float]:
    """outcome_rate() for several possible values in one pass over results
    instead of one pass per value -- what VisualRetrievalEvaluator needs
    for hit/empty/no_image/error together, RetrievalEvaluator needs for
    just failed/not-failed. Same underlying calculation either way; this
    was implemented independently, slightly differently, in both places
    before being pulled out here.
    """
    n = len(results)
    if n == 0:
        return {v: 0.0 for v in possible_values}
    counts = {v: 0 for v in possible_values}
    for r in results:
        value = r.get(outcome_key)
        if value in counts:
            counts[value] += 1
    return {v: counts[v] / n for v in possible_values}