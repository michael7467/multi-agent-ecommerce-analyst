from __future__ import annotations

import json
from pathlib import Path

from app.logging.logger import get_logger
from app.evaluation.eval_history import save_eval_run
from app.services.rag_service import RAGService
from app.evaluation.metrics.retrieval_metrics import (
    KeywordRelevanceJudge,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    average_precision,
    keyword_coverage,
)
from app.evaluation.metrics.reliability_metrics import outcome_rate

logger = get_logger("evaluation.retrieval")


class RetrievalEvaluator:
    def __init__(self, pool_size: int = 200, relevance_judge=None) -> None:
        """pool_size: how many results to fetch per case to approximate
        "every relevant item that exists" for recall/nDCG's denominator.
        Needs to comfortably exceed any single product's review count --
        200 is a starting guess, not a measured value; raise it if a
        product in your eval set has more reviews than that.

        relevance_judge: anything exposing is_relevant(query, review_text,
        relevant_keywords) -> bool. Defaults to KeywordRelevanceJudge (free,
        instant, approximate). Pass LLMRelevanceJudge(...) instead for
        cached LLM-as-judge relevance -- same interface, no other code here
        needs to change.
        """
        self.rag_service = RAGService()
        self.pool_size = pool_size
        self.relevance_judge = relevance_judge or KeywordRelevanceJudge()

    def evaluate_case(
        self,
        product_id: str,
        query: str,
        relevant_keywords: list[str],
        top_k: int = 3,
    ) -> dict:
        base = {"product_id": product_id, "query": query, "top_k": top_k}

        # One fetch, sized to double as both "the top_k being evaluated"
        # and "the full candidate pool" recall/nDCG need as a denominator
        # -- avoids a second round-trip per case.
        try:
            pool = self.rag_service.get_product_evidence(
                product_id=product_id,
                query=query,
                top_k=max(top_k, self.pool_size),
            )
        except Exception:
            # Distinguishable from a genuine zero-match case since the
            # RAGService fix a few turns ago -- before that, this and "no
            # matches" were indistinguishable, which would have made
            # failed_retrieval_rate meaningless (a real outage and a
            # legitimately empty result would have looked identical).
            logger.error(
                f"Retrieval failed for eval case product_id={product_id}, query={query!r}",
                exc_info=True,
            )
            return {
                **base,
                "failed": True,
                "precision_at_k": None, "recall_at_k": None, "mrr": None,
                "ndcg_at_k": None, "context_precision": None, "context_recall": None,
            }

        relevance = [
            self.relevance_judge.is_relevant(query, item.get("review_text", ""), relevant_keywords)
            for item in pool
        ]
        total_relevant = sum(relevance)

        top_k_relevance = relevance[:top_k]
        top_k_texts = [item.get("review_text", "") for item in pool[:top_k]]

        return {
            **base,
            "failed": False,
            "retrieved_count": len(pool),
            "precision_at_k": (sum(top_k_relevance) / top_k) if top_k else 0.0,
            "recall_at_k": recall_at_k(top_k_relevance, total_relevant),
            "mrr": reciprocal_rank(top_k_relevance),
            "ndcg_at_k": ndcg_at_k(top_k_relevance, total_relevant),
            "context_precision": average_precision(top_k_relevance),
            "context_recall": keyword_coverage(top_k_texts, relevant_keywords),
        }

    def run(self, eval_file: str | Path, top_k: int = 3) -> dict:
        path = Path(eval_file)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation file not found: {eval_file}")

        with open(path, "r", encoding="utf-8") as f:
            eval_cases = json.load(f)

        results = [
            self.evaluate_case(
                product_id=case["product_id"],
                query=case["query"],
                relevant_keywords=case["relevant_keywords"],
                top_k=top_k,
            )
            for case in eval_cases
        ]

        n = len(results)
        ok = [r for r in results if not r["failed"]]

        def avg(key: str) -> float | None:
            # recall_at_k can be None (undefined, not zero) when a case
            # has no relevant items at all -- skip those rather than
            # letting them silently drag the average toward 0.
            values = [r[key] for r in ok if r[key] is not None]
            return sum(values) / len(values) if values else None

        summary = {
            "n_cases": n,
            "failed_retrieval_rate": outcome_rate(results, "failed", True),
            "precision_at_k": avg("precision_at_k"),
            "recall_at_k": avg("recall_at_k"),
            "mrr": avg("mrr"),
            "ndcg_at_k": avg("ndcg_at_k"),
            "context_precision": avg("context_precision"),
            "context_recall": avg("context_recall"),
        }

        save_eval_run("retrieval", summary)

        print(f"\n=== Retrieval Evaluation (top_k={top_k}, {n} cases) ===")
        for r in results:
            print(r)
        print("\n=== Summary ===")
        for k, v in summary.items():
            print(f"{k}: {v}")

        return {"results": results, "summary": summary}


def run_retrieval_eval(
    eval_file: str = "data/eval/retrieval_eval_set.json",
    top_k: int = 3,
) -> dict:
    return RetrievalEvaluator().run(eval_file, top_k=top_k)


if __name__ == "__main__":
    run_retrieval_eval(top_k=3)