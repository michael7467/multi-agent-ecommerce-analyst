from __future__ import annotations

from app.evaluation.retrieval_precision_eval import run_retrieval_precision_eval
from app.evaluation.recommendation_eval import run_recommendation_eval


def run_all_advanced_evals() -> None:
    print("\n==============================")
    print("Running Retrieval Precision@K")
    print("==============================")
    run_retrieval_precision_eval(top_k=3)

    print("\n==============================")
    print("Running Recommendation Eval")
    print("==============================")
    run_recommendation_eval()


if __name__ == "__main__":
    run_all_advanced_evals()