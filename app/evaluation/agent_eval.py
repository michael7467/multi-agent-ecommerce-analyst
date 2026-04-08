from __future__ import annotations

from app.agents.orchestrator import Orchestrator


def evaluate_orchestrator(product_id: str, query: str, top_k: int = 3) -> dict:
    orchestrator = Orchestrator()

    try:
        result = orchestrator.run(
            product_id=product_id,
            query=query,
            top_k=top_k,
        )

        final_output = result["final_output"]

        return {
            "success": True,
            "product_id": final_output["product_id"],
            "predicted_class": final_output["predicted_class"],
            "guardrail_status": final_output["guardrail_status"],
            "evidence_count": len(final_output["evidence"]),
            "has_report": bool(final_output["report"].strip()),
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