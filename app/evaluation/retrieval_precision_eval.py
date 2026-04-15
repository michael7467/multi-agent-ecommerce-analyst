from __future__ import annotations

import json
from pathlib import Path

from app.services.rag_service import RAGService


class RetrievalPrecisionEvaluator:
    def __init__(self) -> None:
        self.rag_service = RAGService()

    @staticmethod
    def _is_relevant(review_text: str, relevant_keywords: list[str]) -> bool:
        text = review_text.lower()
        return any(keyword.lower() in text for keyword in relevant_keywords)

    def precision_at_k(
        self,
        product_id: str,
        query: str,
        relevant_keywords: list[str],
        top_k: int = 3,
    ) -> dict:
        evidence = self.rag_service.get_product_evidence(
            product_id=product_id,
            query=query,
            top_k=top_k,
        )

        if not evidence:
            return {
                "product_id": product_id,
                "query": query,
                "top_k": top_k,
                "relevant_count": 0,
                "precision_at_k": 0.0,
            }

        relevant_count = 0
        for item in evidence:
            review_text = item.get("review_text", "")
            if self._is_relevant(review_text, relevant_keywords):
                relevant_count += 1

        precision = relevant_count / top_k

        return {
            "product_id": product_id,
            "query": query,
            "top_k": top_k,
            "relevant_count": relevant_count,
            "precision_at_k": precision,
        }


def run_retrieval_precision_eval(
    eval_file: str = "data/eval/retrieval_eval_set.json",
    top_k: int = 3,
) -> list[dict]:
    path = Path(eval_file)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")

    with open(path, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)

    evaluator = RetrievalPrecisionEvaluator()
    results = []

    for case in eval_cases:
        result = evaluator.precision_at_k(
            product_id=case["product_id"],
            query=case["query"],
            relevant_keywords=case["relevant_keywords"],
            top_k=top_k,
        )
        results.append(result)

    avg_precision = (
        sum(r["precision_at_k"] for r in results) / len(results) if results else 0.0
    )

    print("\n=== Retrieval Precision@K Evaluation ===")
    for r in results:
        print(r)

    print(f"\nAverage Precision@{top_k}: {avg_precision:.4f}")
    return results


if __name__ == "__main__":
    run_retrieval_precision_eval(top_k=3)