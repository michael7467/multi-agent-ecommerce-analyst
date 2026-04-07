from __future__ import annotations

from app.rag.retriever import ReviewRetriever


class RAGService:
    def __init__(self) -> None:
        self.retriever = ReviewRetriever()

    def get_product_evidence(
        self,
        product_id: str,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        results = self.retriever.search_by_product(
            product_id=product_id,
            query=query,
            top_k=top_k,
        )

        evidence = []
        for _, row in results.iterrows():
            evidence.append(
                {
                    "product_id": row["product_id"],
                    "title": row.get("title", ""),
                    "review_text": row.get("review_text", ""),
                    "review_title": row.get("review_title", ""),
                    "categories": row.get("categories", ""),
                    "score": float(row["score"]),
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