from __future__ import annotations

# Verified directly against app/agents/planning_agent.py's _RULES and
# app/agents/dynamic_orchestrator.py's analysis-dict construction -- not
# inferred from flag names.
ALL_FLAGS = [
    "use_competitive", "use_data", "use_buy_decision", "use_topics", "use_trends",
    "use_aspect_sentiment", "use_sentiment", "use_forecast", "use_counterfactuals",
    "use_retrieval", "use_recommender", "use_image_retrieval", "use_summarization",
    "use_report", "use_guardrail", "use_critic",
]

FLAG_TO_OUTPUT_FIELDS: dict[str, list[str]] = {
    "use_competitive": ["competitive_analysis"],
    "use_data": ["title", "categories", "price"],
    "use_buy_decision": ["buy_decision"],
    "use_topics": ["top_themes", "pain_points"],
    "use_trends": ["trend_analysis"],
    "use_aspect_sentiment": ["aspect_sentiment"],
    "use_sentiment": ["sentiment"],
    "use_forecast": ["predicted_class"],
    "use_counterfactuals": ["counterfactuals"],
    "use_retrieval": ["evidence"],
    "use_recommender": ["recommendations"],
    "use_image_retrieval": ["image_similar_products"],
    "use_summarization": ["aspect_summaries"],
    "use_report": ["report"],
    "use_guardrail": ["guardrail_status"],
    "use_critic": ["critic_report", "critic_scores"],
}

# use_guardrail and use_critic only produce output if predicted_class/report
# also exist in dynamic_orchestrator.py (confirmed: their gating conditions
# check those keys explicitly, not just their own flag). A missing field
# for either can be a correct cascading consequence of an earlier step
# being skipped or failing -- not necessarily this flag's own failure.
CASCADING_FLAGS = {"use_guardrail", "use_critic"}


def expand_expected_flags(expected_true: list[str], all_flags: list[str] = ALL_FLAGS) -> dict[str, bool]:
  
    expected_set = set(expected_true)
    unknown = expected_set - set(all_flags)
    if unknown:
        raise ValueError(f"Unknown flag(s) in expected_true: {unknown}")
    return {flag: (flag in expected_set) for flag in all_flags}


def tool_call_precision_recall_f1(predicted: dict[str, bool], expected: dict[str, bool]) -> dict:
  
    flags = set(predicted) | set(expected)
    tp = sum(1 for f in flags if predicted.get(f) and expected.get(f))
    fp = sum(1 for f in flags if predicted.get(f) and not expected.get(f))
    fn = sum(1 for f in flags if not predicted.get(f) and expected.get(f))

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def tool_call_accuracy(predicted: dict[str, bool], expected: dict[str, bool]) -> float:
  
    flags = set(predicted) | set(expected)
    if not flags:
        return 1.0
    matches = sum(1 for f in flags if predicted.get(f) == expected.get(f))
    return matches / len(flags)


def route_exact_match(predicted: dict[str, bool], expected: dict[str, bool]) -> bool:
   
    flags = set(predicted) | set(expected)
    return all(predicted.get(f) is expected.get(f) for f in flags)


def count_active_agents(flags: dict[str, bool]) -> int:
 
    return sum(1 for v in flags.values() if v)


def count_active_agents_from_output(
    final_output: dict,
    flag_to_fields: dict[str, list[str]] = FLAG_TO_OUTPUT_FIELDS,
) -> int:
   
    count = 0
    for fields in flag_to_fields.values():
        if any(final_output.get(field) not in (None, "", [], {}) for field in fields):
            count += 1
    return count


def goal_field_coverage(
    final_output: dict,
    expected_true: list[str],
    flag_to_fields: dict[str, list[str]] = FLAG_TO_OUTPUT_FIELDS,
) -> dict:

    covered, missing, cascading_missing = [], [], []

    for flag in expected_true:
        fields = flag_to_fields.get(flag, [])
        delivered = bool(fields) and all(
            final_output.get(field) not in (None, "", [], {})
            for field in fields
        )
        if delivered:
            covered.append(flag)
        elif flag in CASCADING_FLAGS:
            cascading_missing.append(flag)
        else:
            missing.append(flag)

    total = len(expected_true)
    return {
        "covered": covered,
        "missing": missing,
        "cascading_missing": cascading_missing,
        "goal_accuracy": len(covered) / total if total else 1.0,
    }
