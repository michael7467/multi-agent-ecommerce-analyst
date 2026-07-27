from __future__ import annotations

from app.evaluation.evaluators.retrieval_evaluator import run_retrieval_eval
from app.evaluation.evaluators.recommendation_evaluator import run_recommendation_eval


def run_all_advanced_evals() -> None:
    print("\n==============================")
    print("Running Retrieval Precision@K")
    print("==============================")
    run_retrieval_eval(top_k=3)

    print("\n==============================")
    print("Running Recommendation Eval")
    print("==============================")
    run_recommendation_eval()


if __name__ == "__main__":
    run_all_advanced_evals()