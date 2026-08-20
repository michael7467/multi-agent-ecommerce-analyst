from __future__ import annotations


from app.agents.langgraph_orchestrator import LangGraphOrchestrator


def evaluate_orchestrator(
    product_id: str,
    query: str,
    top_k: int = 3,
    precomputed_result: dict | None = None,
) -> dict:
    """precomputed_result: pass an already-computed orchestrator.run()
    result to skip running the pipeline again. run_all_eval.py needs this
    result anyway for report_eval, and the full pipeline can mean up to 16
    sequential LLM calls -- running it twice for the same inputs just to
    get two views of the same answer is a real, avoidable cost, not a
    trivial one.
    """
    try:
        if precomputed_result is None:
            orchestrator = LangGraphOrchestrator()
            precomputed_result = orchestrator.run(
                product_id=product_id,
                query=query,
                top_k=top_k,
            )

        final_output = precomputed_result["final_output"]

        return {
            "success": True,
            "product_id": final_output.get("product_id"),
            "predicted_class": final_output.get("predicted_class"),
            "guardrail_status": final_output.get("guardrail_status"),
            "evidence_count": len(final_output.get("evidence", [])),
            "has_report": bool(final_output.get("report", "").strip()),
            "failed_steps": final_output.get("failed_steps", []),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


if __name__ == "__main__":
    result = evaluate_orchestrator(
        product_id="B09SPZPDJK",
        query="sound quality and noise cancellation",
        top_k=3,
    )

    print("Agent Evaluation Result:")
    print(result)