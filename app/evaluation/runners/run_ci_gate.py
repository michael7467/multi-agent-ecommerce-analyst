from __future__ import annotations

import sys

from app.evaluation.runners.run_all_eval import run_all
from app.evaluation.regression_check import find_regressions
from app.evaluation.eval_history import save_eval_run, load_eval_history

# Maps run_all()'s actual result keys to eval_history.py's real,
# constrained evaluator_type values (exactly "retrieval", "agentic",
# "visual", "generation" -- confirmed directly, not assumed). agent_eval
# and report_eval are deliberately excluded, not silently mapped to a
# made-up type: eval_history.py has no evaluator_type for either, so
# there's nothing to persist or compare them against. If those two need
# regression tracking later, that's a real schema change to
# eval_history.py itself, not something to work around here.
_RESULT_KEY_TO_EVALUATOR_TYPE = {
    "rag_eval": "retrieval",
    "generation_eval": "generation",
    "agentic_eval": "agentic",
}


def run_ci_gate(
    product_id: str,
    query: str,
    top_k: int = 3,
    relevant_keywords: list[str] | None = None,
    expected_true: list[str] | None = None,
) -> int:
    """
    CI entry point. Returns a process exit code (0 = pass, 1 = fail) --
    the actual mechanism that lets a GitHub Actions step, and therefore
    branch protection, block a merge on regression.

    IMPORTANT, real operational constraint: eval_history.py's store has
    no product_id/query awareness at all -- "the previous run" means
    only "the most recent run of this evaluator_type," full stop. This
    function's comparisons are only meaningful if CI always calls this
    with the SAME product_id/query every time. Calling this ad hoc with
    a different product/query would silently corrupt the next real CI
    run's baseline -- confirmed as a real risk by reading eval_history.py
    directly, not a hypothetical concern.
    """
    current = run_all(
        product_id=product_id,
        query=query,
        top_k=top_k,
        relevant_keywords=relevant_keywords,
        expected_true=expected_true,
    )

    all_regressions: list[str] = []

    for result_key, evaluator_type in _RESULT_KEY_TO_EVALUATOR_TYPE.items():
        current_summary = current.get(result_key)
        if current_summary is None:
            continue

        previous_runs = load_eval_history(evaluator_type=evaluator_type, limit=1)

        if not previous_runs:
            print(f"No previous '{evaluator_type}' run found -- nothing to compare, treating as a pass.")
        else:
            previous_summary = previous_runs[0]["summary"]
            regressions = find_regressions(previous_summary, current_summary)
            if regressions:
                all_regressions.extend(f"[{evaluator_type}] {r}" for r in regressions)

        # Saved regardless of pass/fail, same reasoning as before -- this
        # run becomes the next comparison's baseline either way.
        # save_eval_run never raises (confirmed directly in
        # eval_history.py), so a history-write failure can't take this
        # script down.
        save_eval_run(evaluator_type=evaluator_type, summary=current_summary)

    # agent_eval and report_eval are real parts of the result, still
    # printed for visibility, just not persisted or regression-checked --
    # no evaluator_type exists for either in the current schema.
    print("\n(agent_eval and report_eval are not regression-checked -- no matching evaluator_type in eval_history.py's current schema)")
    print(f"agent_eval: {current.get('agent_eval')}")
    print(f"report_eval: {current.get('report_eval')}")

    if all_regressions:
        print("\nREGRESSIONS DETECTED:")
        for r in all_regressions:
            print(f"  - {r}")
        return 1

    print("\nNo regressions detected.")
    return 0


if __name__ == "__main__":
    exit_code = run_ci_gate(
        product_id="B09SPZPDJK",
        query="sound quality and noise cancellation",
        top_k=3,
        relevant_keywords=["sound", "audio", "noise", "cancellation", "cancelling"],
        expected_true=["use_sentiment", "use_retrieval", "use_aspect_sentiment", "use_summarization"],
    )
    sys.exit(exit_code)