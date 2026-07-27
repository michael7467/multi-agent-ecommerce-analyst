from __future__ import annotations


def classify_visual_retrieval_outcome(
    retrieved_count: int | None,
    exception: Exception | None = None,
) -> str:
    """Classifies a single visual retrieval attempt into one of four
    outcomes, given either a result count (success) or a caught exception
    (failure). Pulled out of VisualRetrievalEvaluator.evaluate_product() so
    the classification decision -- as opposed to the actual retrieval call,
    which is inherently I/O -- can be tested without a real or mocked
    ImageRetriever at all.

    - "hit": search ran, found at least one similar product.
    - "empty": search ran fine, found nothing -- a real, if uninteresting,
      outcome, distinct from every failure case below.
    - "no_image": the product isn't in the image metadata at all. This is
      ImageRetriever.search_by_product()'s specific documented failure mode
      ("Product not found in image metadata: ..."), not a generic error --
      it means there's no image for this product, not that anything broke.
    - "error": a genuine, unexpected failure -- anything else.

    retrieved_count is ignored when exception is given (there's nothing to
    count if the call never returned).
    """
    if exception is not None:
        if isinstance(exception, ValueError) and "not found in image metadata" in str(exception):
            return "no_image"
        return "error"

    return "hit" if (retrieved_count or 0) > 0 else "empty"