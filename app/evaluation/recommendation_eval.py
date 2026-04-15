from __future__ import annotations

import json
from pathlib import Path

from app.agents.orchestrator import Orchestrator


class RecommendationEvaluator:
    def __init__(self) -> None:
        self.orchestrator = Orchestrator()

    @staticmethod
    def simple_recommendation_rule(final_output: dict) -> str:
        predicted_class = final_output.get("predicted_class", "")
        sentiment = final_output.get("sentiment", {})

        positive_ratio = float(sentiment.get("positive_review_ratio", 0.0))
        negative_ratio = float(sentiment.get("negative_review_ratio", 0.0))

        if predicted_class == "high" and negative_ratio > 0.25:
            return "wait"

        if predicted_class == "high" and positive_ratio >= 0.60:
            return "buy_now"

        if predicted_class == "mid" and positive_ratio >= 0.60:
            return "buy_now"

        return "wait"

    def evaluate_case(self, product_id: str, query: str, expected_decision: str) -> dict:
        result = self.orchestrator.run(product_id=product_id, query=query, top_k=3)
        final_output = result["final_output"]

        predicted_decision = self.simple_recommendation_rule(final_output)
        is_correct = predicted_decision == expected_decision

        return {
            "product_id": product_id,
            "query": query,
            "expected_decision": expected_decision,
            "predicted_decision": predicted_decision,
            "correct": is_correct,
        }


def run_recommendation_eval(
    eval_file: str = "data/eval/recommendation_eval_set.json",
) -> list[dict]:
    path = Path(eval_file)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")

    with open(path, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)

    evaluator = RecommendationEvaluator()
    results = []

    for case in eval_cases:
        result = evaluator.evaluate_case(
            product_id=case["product_id"],
            query=case["query"],
            expected_decision=case["expected_decision"],
        )
        results.append(result)

    accuracy = (
        sum(1 for r in results if r["correct"]) / len(results) if results else 0.0
    )

    print("\n=== Recommendation Evaluation ===")
    for r in results:
        print(r)

    print(f"\nRecommendation Accuracy: {accuracy:.4f}")
    return results


if __name__ == "__main__":
    run_recommendation_eval()