from __future__ import annotations

import json
from pathlib import Path

from app.logging.logger import get_logger
from app.evaluation.eval_history import save_eval_run
from app.rag.image_retriever import ImageRetriever
from app.evaluation.metrics.visual_metrics import classify_visual_retrieval_outcome
from app.evaluation.metrics.reliability_metrics import outcome_rates

logger = get_logger("evaluation.visual_retrieval")


class VisualRetrievalEvaluator:
    """Visual retrieval hit rate, and optionally image-text alignment.

    Hit rate needs nothing but product_ids: across a set of them, how
    often does image-based similar-product retrieval actually return
    results. This is image-to-image retrieval (find products that LOOK
    like this one) -- ImageRetriever.search_by_product() doesn't take a
    text query, so hit rate alone can't say anything about whether the
    results are good matches to a query, only whether the pipeline works.

    Passing a vision_judge and a query per product adds image-text
    alignment on top: for each retrieved product's image, does it
    actually match what that query was asking about, as judged by a
    vision-capable LLM looking at the real image. This is what actually
    answers "are these good matches", which hit rate alone can't.
    """

    def __init__(self, vision_judge=None) -> None:
        self.retriever = ImageRetriever()
        self.vision_judge = vision_judge

    def evaluate_product(self, product_id: str, top_k: int = 5, query: str | None = None) -> dict:
        try:
            results = self.retriever.search_by_product(product_id=product_id, top_k=top_k)
        except Exception as e:
            if not isinstance(e, ValueError) or "not found in image metadata" not in str(e):
                logger.error(f"Visual retrieval failed for product_id={product_id}", exc_info=True)
            outcome = classify_visual_retrieval_outcome(retrieved_count=None, exception=e)
            return {"product_id": product_id, "outcome": outcome, "retrieved_count": 0}

        count = len(results)
        outcome = classify_visual_retrieval_outcome(retrieved_count=count, exception=None)
        record = {"product_id": product_id, "outcome": outcome, "retrieved_count": count}

        # Alignment scoring is opt-in per call: needs both a query (what
        # is this retrieval even for) and a judge (something that can
        # actually look at the images). Without either, this is silently
        # skipped rather than computed as some default -- there's nothing
        # honest to compute it from.
        if query and self.vision_judge and count > 0:
            aligned_flags = []
            for _, row in results.iterrows():
                image_url = row.get("image_url", "")
                if not image_url:
                    continue
                aligned_flags.append(self.vision_judge.is_relevant(query, image_url))

            if aligned_flags:
                record["image_text_alignment"] = sum(aligned_flags) / len(aligned_flags)
                record["images_judged"] = len(aligned_flags)

        return record

    def run(self, cases: list[str | dict], top_k: int = 5) -> dict:
        """cases: either a list of bare product_id strings (hit rate only),
        or a list of {"product_id": ..., "query": ...} dicts (hit rate +
        image-text alignment, if this evaluator was built with a
        vision_judge). Mixed lists are fine -- entries without a "query"
        key just skip alignment scoring for that one product.
        """
        results = []
        for case in cases:
            if isinstance(case, str):
                results.append(self.evaluate_product(case, top_k=top_k))
            else:
                results.append(self.evaluate_product(case["product_id"], top_k=top_k, query=case.get("query")))

        n = len(results)
        rates = outcome_rates(results, "outcome", ["hit", "empty", "no_image", "error"])

        alignment_scores = [r["image_text_alignment"] for r in results if "image_text_alignment" in r]

        summary = {
            "n_products": n,
            "hit_rate": rates["hit"],
            "empty_rate": rates["empty"],
            "no_image_rate": rates["no_image"],
            "error_rate": rates["error"],
            "avg_image_text_alignment": (
                sum(alignment_scores) / len(alignment_scores) if alignment_scores else None
            ),
        }

        save_eval_run("visual", summary)

        print(f"\n=== Visual Retrieval Evaluation (top_k={top_k}, {n} products) ===")
        for r in results:
            print(r)
        print("\n=== Summary ===")
        for k, v in summary.items():
            print(f"{k}: {v}")

        return {"results": results, "summary": summary}


def run_visual_retrieval_eval(
    product_ids_file: str = "data/eval/visual_eval_product_ids.json",
    top_k: int = 5,
    with_vision_judge: bool = False,
) -> dict:
    path = Path(product_ids_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Eval file not found: {product_ids_file} -- expects either a "
            f"JSON list of product_id strings, or a list of "
            f'{{"product_id": ..., "query": ...}} objects for alignment scoring.'
        )

    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    vision_judge = None
    if with_vision_judge:
        from app.evaluation.judges.vision_relevance_judge import VisionRelevanceJudge
        vision_judge = VisionRelevanceJudge()

    return VisualRetrievalEvaluator(vision_judge=vision_judge).run(cases, top_k=top_k)


if __name__ == "__main__":
    run_visual_retrieval_eval(top_k=5)