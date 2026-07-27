from __future__ import annotations

import pandas as pd
import streamlit as st

from app.evaluation.eval_history import load_eval_history


st.set_page_config(
    page_title="Evaluation History",
    page_icon="\U0001F4CA",
    layout="wide",
)

# Which summary keys are worth charting, per evaluator type -- these
# genuinely don't share a schema (confirmed against the real summary dicts
# each evaluator's run() produces), so this has to be per-type, not generic.
CHART_METRICS: dict[str, list[str]] = {
    "retrieval": ["precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "context_precision", "context_recall"],
    "agentic": ["avg_tool_call_accuracy", "avg_tool_call_precision", "avg_tool_call_recall", "avg_tool_call_f1", "route_exact_match_rate"],
    "visual": ["hit_rate", "empty_rate", "no_image_rate", "error_rate", "avg_image_text_alignment"],
    "generation": ["avg_faithfulness", "avg_hallucination_rate", "response_relevancy_rate", "avg_critic_derived_faithfulness"],
}

TYPE_LABELS: dict[str, str] = {
    "retrieval": "Retrieval",
    "agentic": "Agentic (Routing)",
    "visual": "Multimodal",
    "generation": "Generation",
}

RATE_HINT = (
    "Rates and scores here are generally 0-1 -- 1.0 isn't automatically "
    "\"perfect,\" it means every case in that run hit the target; check the "
    "case count above before reading too much into a single run."
)


def render_history_for_type(evaluator_type: str) -> None:
    st.subheader(TYPE_LABELS.get(evaluator_type, evaluator_type))

    runs = load_eval_history(evaluator_type=evaluator_type, limit=200)
    if not runs:
        st.info(
            f"No {TYPE_LABELS.get(evaluator_type, evaluator_type).lower()} runs recorded yet. "
            f"Run `python -m app.evaluation.evaluators.{'visual_retrieval_evaluator' if evaluator_type == 'visual' else evaluator_type + '_evaluator'}` "
            "to see results appear here."
        )
        return

    # Oldest-to-newest for charting (a line chart reading right-to-left as
    # "backwards in time" is confusing); table stays newest-first, since
    # that's what you want scanning a list of runs.
    rows = list(reversed(runs))

    table_rows = []
    for run in runs:
        row = {"Run Time (UTC)": run["created_at"][:19].replace("T", " ")}
        row.update(run["summary"])
        table_rows.append(row)
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    metrics = CHART_METRICS.get(evaluator_type, [])
    available_metrics = [
        m for m in metrics
        if any(r["summary"].get(m) is not None for r in rows)
    ]

    if len(rows) < 2:
        st.caption("Need at least 2 runs to show a trend line -- only 1 recorded so far.")
    elif not available_metrics:
        st.caption("No chartable numeric metrics in these runs yet (all None so far).")
    else:
        chart_df = pd.DataFrame(
            {m: [r["summary"].get(m) for r in rows] for m in available_metrics},
            index=pd.to_datetime([r["created_at"] for r in rows]),
        )
        st.line_chart(chart_df)
        st.caption(RATE_HINT)


def main() -> None:
    st.title("\U0001F4CA Evaluation History")
    st.markdown(
        "Results saved automatically each time one of the four batch "
        "evaluators runs (`retrieval_evaluator`, `agentic_evaluator`, "
        "`visual_retrieval_evaluator`, `generation_evaluator`). "
        "This is separate from the live Grafana dashboard -- those track "
        "real production traffic continuously; this tracks periodic, "
        "eval-set-driven runs over time, the kind you'd run before a "
        "deploy or after changing a prompt, not on every request."
    )

    st.markdown("---")

    all_types = list(TYPE_LABELS.keys())
    selected = st.multiselect(
        "Show evaluator types",
        options=all_types,
        default=all_types,
        format_func=lambda t: TYPE_LABELS.get(t, t),
    )

    if not selected:
        st.info("Select at least one evaluator type above to see its history.")
        return

    for evaluator_type in selected:
        render_history_for_type(evaluator_type)
        st.markdown("---")


if __name__ == "__main__":
    main()
