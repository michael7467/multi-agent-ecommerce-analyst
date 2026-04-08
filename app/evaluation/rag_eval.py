from __future__ import annotations

from app.services.rag_service import RAGService


def evaluate_product_retrieval(product_id: str, query: str, top_k: int = 3) -> dict:
    service = RAGService()
    evidence = service.get_product_evidence(
        product_id=product_id,
        query=query,
        top_k=top_k,
    )

    retrieved_items = len(evidence)
    non_empty_reviews = sum(1 for ev in evidence if ev.get("review_text", "").strip())
    avg_similarity_score = (
        sum(ev["score"] for ev in evidence) / retrieved_items if retrieved_items > 0 else 0.0
    )

    return {
        "product_id": product_id,
        "query": query,
        "retrieved_items": retrieved_items,
        "non_empty_reviews": non_empty_reviews,
        "avg_similarity_score": avg_similarity_score,
    }


if __name__ == "__main__":
    result = evaluate_product_retrieval(
        product_id="B09SPZPDJK",
        query="sound quality and noise cancellation",
        top_k=3,
    )

    print("RAG Evaluation Result:")
    print(result)