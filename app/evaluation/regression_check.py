from __future__ import annotations


def find_regressions(baseline: dict, current: dict, tolerance: float = 0.05, _path: str = "") -> list[str]:
    """
    Recursively walks baseline and current in parallel, comparing any
    matching numeric values at the same key path. Returns a human-readable
    description for each metric that dropped by more than `tolerance`.

    Deliberately generic rather than hardcoded to specific metric paths --
    doesn't need to know the internal shape of agent_eval/rag_eval/
    report_eval/generation_eval/agentic_eval individually, and won't
    silently stop working if any of those evaluators add or rename a
    metric later.

    ASSUMPTION, stated rather than hidden: this assumes higher is always
    better for every numeric metric compared. True for what's been
    confirmed this session (hallucination_risk is explicitly 10=low-risk,
    i.e. already higher-is-better), but not verified against every metric
    in all 5 evaluators, since their exact output shapes haven't all been
    seen. A metric where lower is genuinely better (e.g. a raw latency
    number) would be silently mishandled by this function as-is.

    Booleans are skipped, not compared numerically -- bool is a subclass
    of int in Python, so this has to be checked explicitly or a
    True->False "skipped" flag flip would be nonsensically reported as a
    huge regression.
    """
    regressions: list[str] = []

    for key, current_value in current.items():
        full_path = f"{_path}.{key}" if _path else key
        baseline_value = baseline.get(key) if isinstance(baseline, dict) else None

        if isinstance(current_value, dict) and isinstance(baseline_value, dict):
            regressions.extend(find_regressions(baseline_value, current_value, tolerance, full_path))
            continue

        if isinstance(current_value, bool) or isinstance(baseline_value, bool):
            continue

        if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
            drop = baseline_value - current_value
            if drop > tolerance:
                regressions.append(
                    f"{full_path}: {baseline_value:.4f} -> {current_value:.4f} (dropped {drop:.4f})"
                )

    return regressions
