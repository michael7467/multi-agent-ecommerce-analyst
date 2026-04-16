from __future__ import annotations

from app.rag.qdrant_retriever import QdrantRetriever


class RAGService:
    def __init__(self) -> None:
        self.retriever = QdrantRetriever()

    def get_product_evidence(
        self,
        product_id: str,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        results = self.retriever.search(
            query=query,
            top_k=top_k,
            product_id=product_id,
        )

        evidence = []
        for item in results:
            evidence.append(
                {
                    "product_id": item.get("product_id", ""),
                    "title": item.get("title", ""),
                    "review_text": item.get("review_text", ""),
                    "review_title": item.get("review_title", ""),
                    "categories": item.get("categories", ""),
                    "score": float(item.get("score", 0.0)),
                }
            )

        return evidence


if __name__ == "__main__":
    service = RAGService()

    sample_product_id = "B09SPZPDJK"
    query = "sound quality and noise cancellation"

    evidence = service.get_product_evidence(
        product_id=sample_product_id,
        query=query,
        top_k=3,
    )

    for i, item in enumerate(evidence, start=1):
        print(f"\nEvidence {i}")
        print(item)