from __future__ import annotations

import json
from pathlib import Path

from app.logging.logger import get_logger
from app.evaluation.eval_history import save_eval_run
from app.agents.langgraph_orchestrator import LangGraphOrchestrator
from app.services.report_service import ReportService
from app.evaluation.judges.faithfulness_judge import FaithfulnessJudge
from app.evaluation.judges.llm_relevance_judge import LLMRelevanceJudge
from app.evaluation.metrics.generation_metrics import (
    faithfulness_from_critic_scores,
    hallucination_rate_from_critic_scores,
)
from app.evaluation.metrics.reliability_metrics import outcome_rate

logger = get_logger("evaluation.generation")


class GenerationEvaluator:
    """Generation-quality metrics for a report, checked against its query
    and the analysis_result it was generated from.

    Not all five commonly-requested generation metrics get a distinct
    implementation -- some are the same underlying measurement under
    different names for this specific system, and building separate
    pipelines for each would be the same "one idea, several independent
    copies" problem already found and fixed three times this session
    (guardrail's alignment check, report_service's, report_eval's).

    - faithfulness / hallucination_rate / factual_correctness all come
      from ONE claim-level grounding check (FaithfulnessJudge). There's
      no separate notion of "factual correctness" beyond "grounded in
      analysis_result" for this domain -- no external knowledge base
      exists to check facts against beyond what the pipeline itself
      produced.
    - response_relevancy reuses LLMRelevanceJudge directly, unchanged --
      it already does exactly this (is text X relevant to query Y), just
      previously pointed at review text instead of report text.
    - answer_accuracy is deliberately NOT implemented as a new metric
      here. RAGAS's definition compares against a reference/ground-truth
      answer, which doesn't exist for this system's open-ended report
      generation -- there's no single "correct" report for a subjective
      product analysis. The one place this system has a genuine,
      checkable reference is predicted_class, and that's already covered
      by class_alignment.py / check_report_alignment. Not duplicating it.

    Also reports a second, free faithfulness signal derived from
    CriticAgent's existing hallucination_risk score (critic_scores, if
    the request had use_critic enabled) -- zero extra LLM calls, since
    that score already exists wherever critic ran. It's a coarser signal
    than FaithfulnessJudge's per-claim check (one holistic number, not
    "which claim specifically"), and only available when use_critic was
    on for that request, but costs nothing when it is.
    """

    def __init__(self) -> None:
   
        self._orchestrator = None
        self.report_service = ReportService()
        self.faithfulness_judge = FaithfulnessJudge()
        self.relevance_judge = LLMRelevanceJudge()

    @property
    def orchestrator(self) -> LangGraphOrchestrator:
        if self._orchestrator is None:
            self._orchestrator = LangGraphOrchestrator()
        return self._orchestrator

    def evaluate(self, analysis_result: dict, report: str, query: str) -> dict:

        context = self.report_service._build_prompt(analysis_result)

        faithfulness_result = self.faithfulness_judge.check(context, report)

        try:
            is_relevant = self.relevance_judge.is_relevant(query, report)
        except Exception:
            logger.error(f"Relevance judgment failed for query={query!r}", exc_info=True)
            is_relevant = None

        result = {
            "faithfulness": faithfulness_result["faithfulness"],
            "hallucination_rate": faithfulness_result["hallucination_rate"],
            "factual_correctness": faithfulness_result["faithfulness"],
            "n_claims_checked": faithfulness_result["n_claims"],
            "ungrounded_claims": [
                c["claim"] for c in faithfulness_result["claims"] if not c["grounded"]
            ],
            "response_relevancy": is_relevant,
        }

        if "error" in faithfulness_result:
            result["faithfulness_error"] = faithfulness_result["error"]

        critic_scores = analysis_result.get("critic_scores")
        if critic_scores:
            result["critic_derived_faithfulness"] = faithfulness_from_critic_scores(critic_scores)
            result["critic_derived_hallucination_rate"] = hallucination_rate_from_critic_scores(critic_scores)

        return result

    def run(self, eval_file: str | Path, top_k: int = 3) -> dict:
        """eval_file entries: {"product_id": ..., "query": ...}. Runs the
        full pipeline for each case -- needed to get a real report and
        analysis_result to actually check, not something worth
        hand-crafting separately in a test file.
        """
        path = Path(eval_file)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation file not found: {eval_file}")

        with open(path, "r", encoding="utf-8") as f:
            cases = json.load(f)

        results = []
        for case in cases:
            base = {"product_id": case["product_id"], "query": case["query"]}
            try:
                orchestrator_result = self.orchestrator.run(
                    product_id=case["product_id"], query=case["query"], top_k=top_k,
                )
                final_output = orchestrator_result["final_output"]
            except Exception:
                logger.error(f"Orchestrator run failed for product_id={case['product_id']}", exc_info=True)
                results.append({**base, "failed": True, "reason": "orchestrator_run_failed"})
                continue

            report = final_output.get("report")
            if not report:
          
                results.append({**base, "failed": True, "reason": "no_report_in_output"})
                continue

            metrics = self.evaluate(final_output, report, case["query"])
            results.append({**base, "failed": False, **metrics})

        n = len(results)
        ok = [r for r in results if not r["failed"]]

        def avg(key: str):
            values = [r[key] for r in ok if r.get(key) is not None]
            return sum(values) / len(values) if values else None

        summary = {
            "n_cases": n,
            "failed_rate": outcome_rate(results, "failed", True),
            "avg_faithfulness": avg("faithfulness"),
            "avg_hallucination_rate": avg("hallucination_rate"),
            "response_relevancy_rate": (
                sum(1 for r in ok if r.get("response_relevancy") is True) / len(ok) if ok else None
            ),
            "avg_critic_derived_faithfulness": avg("critic_derived_faithfulness"),
        }

        save_eval_run("generation", summary)

        print(f"\n=== Generation Evaluation ({n} cases) ===")
        for r in results:
            print(r)
        print("\n=== Summary ===")
        for k, v in summary.items():
            print(f"{k}: {v}")

        return {"results": results, "summary": summary}


def run_generation_eval(
    eval_file: str = "data/eval/generation_eval_set.json",
    top_k: int = 3,
) -> dict:
    return GenerationEvaluator().run(eval_file, top_k=top_k)


if __name__ == "__main__":
    run_generation_eval()