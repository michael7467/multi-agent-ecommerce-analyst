from __future__ import annotations

from app.evaluation.agent_eval import evaluate_orchestrator
from app.evaluation.rag_eval import evaluate_product_retrieval
from app.evaluation.report_eval import check_report_alignment
from app.agents.orchestrator import Orchestrator


def run_all(product_id: str, query: str, top_k: int = 3) -> dict:
    orchestrator = Orchestrator()
    result = orchestrator.run(product_id=product_id, query=query, top_k=top_k)
    final_output = result["final_output"]

    agent_result = evaluate_orchestrator(product_id=product_id, query=query, top_k=top_k)
    rag_result = evaluate_product_retrieval(product_id=product_id, query=query, top_k=top_k)
    report_result = check_report_alignment(
        predicted_class=final_output["predicted_class"],
        report=final_output["report"],
    )

    return {
        "agent_eval": agent_result,
        "rag_eval": rag_result,
        "report_eval": report_result,
    }


if __name__ == "__main__":
    results = run_all(
        product_id="B09SPZPDJK",
        query="sound quality and noise cancellation",
        top_k=3,
    )

    print("\n=== ALL EVALUATIONS ===")
    for name, value in results.items():
        print(f"\n{name}:")
        print(value)