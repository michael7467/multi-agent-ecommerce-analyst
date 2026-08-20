from __future__ import annotations

from app.agents.langgraph_orchestrator import LangGraphOrchestrator
from app.evaluation.evaluators.agent_evaluator import evaluate_orchestrator
from app.evaluation.evaluators.rag_evaluator import evaluate_product_retrieval
from app.evaluation.evaluators.report_evaluator import check_report_alignment
from app.evaluation.evaluators.retrieval_evaluator import RetrievalEvaluator
from app.evaluation.evaluators.generation_evaluator import GenerationEvaluator
from app.evaluation.evaluators.agentic_evaluator import AgenticEvaluator


def run_all(
    product_id: str,
    query: str,
    top_k: int = 3,
    relevant_keywords: list[str] | None = None,
    expected_true: list[str] | None = None,
) -> dict:


    orchestrator = LangGraphOrchestrator()
    orchestrator_result = orchestrator.run(product_id=product_id, query=query, top_k=top_k)
    final_output = orchestrator_result["final_output"]

    agent_result = evaluate_orchestrator(
        product_id=product_id,
        query=query,
        top_k=top_k,
        precomputed_result=orchestrator_result,
    )

    rag_result = evaluate_product_retrieval(product_id=product_id, query=query, top_k=top_k)

    if relevant_keywords:
        rag_result["advanced_metrics"] = RetrievalEvaluator().evaluate_case(
            product_id=product_id,
            query=query,
            relevant_keywords=relevant_keywords,
            top_k=top_k,
        )


    predicted_class = final_output.get("predicted_class")
    report = final_output.get("report")

    if predicted_class is not None and report:
        report_result = check_report_alignment(predicted_class=predicted_class, report=report)
    else:
        report_result = {
            "skipped": True,
            "reason": "no predicted_class and/or report in this analysis "
                      "(forecasting or reporting wasn't enabled for this query)",
        }


    if report:
        generation_result = GenerationEvaluator().evaluate(final_output, report, query)
    else:
        generation_result = {"skipped": True, "reason": "no report in this analysis"}


    if expected_true:
        agentic_result = AgenticEvaluator().evaluate_routing(query, expected_true)
    else:
        agentic_result = {"skipped": True, "reason": "no expected_true flags given"}

    return {
        "agent_eval": agent_result,
        "rag_eval": rag_result,
        "report_eval": report_result,
        "generation_eval": generation_result,
        "agentic_eval": agentic_result,
    }


if __name__ == "__main__":
    results = run_all(
        product_id="B09SPZPDJK",
        query="sound quality and noise cancellation",
        top_k=3,
        relevant_keywords=["sound", "audio", "noise", "cancellation", "cancelling"],
        expected_true=["use_sentiment", "use_retrieval", "use_aspect_sentiment", "use_summarization"],
    )

    print("\n=== ALL EVALUATIONS ===")
    for name, value in results.items():
        print(f"\n{name}:")
        print(value)