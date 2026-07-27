from __future__ import annotations


def faithfulness_from_critic_scores(critic_scores: dict) -> float | None:
    """Converts CriticAgent's hallucination_risk (1-10, 10 = low risk)
    into a 0-1 faithfulness score (1 = fully faithful) -- the more common
    scale for this metric elsewhere.

    Deliberately not a new LLM call: CriticAgent already asks this exact
    question in production (see critic_agent.py's prompt), and a second,
    independent judge asking it again would repeat the same mistake fixed
    three times already this session (GuardrailAgent, ReportService, and
    report_eval.py all independently re-implemented the same predicted-
    class alignment check before being consolidated).

    Returns None, not 0.0, if hallucination_risk wasn't parseable from the
    critic's response -- a missing score means the LLM didn't follow the
    requested format, not that faithfulness is actually zero. See
    critic_agent.py's _parse_critique.
    """
    risk = critic_scores.get("hallucination_risk")
    if risk is None:
        return None
    return max(0.0, min(1.0, risk / 10.0))


def hallucination_rate_from_critic_scores(critic_scores: dict) -> float | None:
    """1 - faithfulness. Same underlying source and question as
    faithfulness_from_critic_scores, read in the inverse direction --
    these are two names for one measurement, not two independent signals.
    """
    faithfulness = faithfulness_from_critic_scores(critic_scores)
    return None if faithfulness is None else (1.0 - faithfulness)
