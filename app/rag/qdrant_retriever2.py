from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import re
from qdrant_client import models
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.config.settings import settings
from app.logging.logger import get_logger
from app.observability.metrics import (
    RETRIEVAL_LATENCY_SECONDS,
    RETRIEVAL_REQUESTS_TOTAL,
)
from app.observability.tracing import get_tracer
from app.rag.qdrant_client_manager import get_qdrant_client

logger = get_logger("rag.qdrant")

COLLECTION_NAME = settings.qdrant_collection_name
EMBED_MODEL_NAME = settings.embedding_model_name
RERANK_MODEL_NAME = getattr(settings, "rerank_model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")

_MODEL: SentenceTransformer | None = None
_RERANK_MODEL: CrossEncoder | None = None


class QdrantRetriever:
    def __init__(self) -> None:
        global _MODEL, _RERANK_MODEL

        self.client = get_qdrant_client()
        self.tracer = get_tracer("app.qdrant_retriever")

        # Embedding model (singleton)
        if _MODEL is None:
            logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
            _MODEL = SentenceTransformer(EMBED_MODEL_NAME)
        self.model = _MODEL

        # CrossEncoder rerank model (singleton)
        if _RERANK_MODEL is None:
            logger.info(f"Loading rerank model: {RERANK_MODEL_NAME}")
            _RERANK_MODEL = CrossEncoder(RERANK_MODEL_NAME)
        self.rerank_model = _RERANK_MODEL

        if not self.client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Qdrant collection '{COLLECTION_NAME}' does not exist. "
                "Build/upload the Qdrant index before running retrieval."
            )

    # -----------------------------
    # Core helpers
    # -----------------------------

    def _embed_query(self, query: str) -> list[float]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("QdrantRetriever: query must be a non-empty string")

        vector = self.model.encode(
            [query.strip()],
            normalize_embeddings=True,
        )[0]

        return vector.astype(np.float32).tolist()

    def _build_filter(
        self,
        product_id: str | None = None,
        semantic_filters: dict[str, Iterable[str]] | None = None,
    ) -> models.Filter | None:
        must_conditions: list[models.FieldCondition] = []

        if product_id:
            must_conditions.append(
                models.FieldCondition(
                    key="product_id",
                    match=models.MatchValue(value=str(product_id)),
                )
            )

        if semantic_filters:
            for key, values in semantic_filters.items():
                # semantic filters: match any of the provided values
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=[str(v) for v in values]),
                    )
                )

        if not must_conditions:
            return None

        return models.Filter(must=must_conditions)

    # -----------------------------
    # Hybrid scoring (text + vector)
    # -----------------------------

    def _keyword_score(self, text: str, query: str) -> float:
        # simple keyword overlap score
        query_tokens = [t for t in re.split(r"\W+", query.lower()) if t]
        if not query_tokens:
            return 0.0

        text_lower = text.lower()
        hits = sum(1 for t in query_tokens if re.search(r"\b" + re.escape(t) + r"\b", text_lower))
        return hits / len(query_tokens)

    def _hybrid_score(self, vector_score: float, keyword_score: float, alpha: float = 0.7) -> float:
        # alpha: weight for vector score, (1-alpha) for keyword score
        return alpha * vector_score + (1.0 - alpha) * keyword_score

    # -----------------------------
    # Reranking with CrossEncoder
    # -----------------------------

    def _rerank(self, query: str, results: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not results:
            return results

        pairs = []
        for r in results:
            # use review_text + description as document text
            doc_text = " ".join(
                [
                    r.get("review_title", "") or "",
                    r.get("review_text", "") or "",
                    r.get("title", "") or "",
                    r.get("description", "") or "",
                ]
            ).strip()
            pairs.append((query, doc_text))

        scores = self.rerank_model.predict(pairs)
        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)

        # sort by rerank_score desc
        results_sorted = sorted(results, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return results_sorted[:top_k]

    # -----------------------------
    # Dynamic top_k selection
    # -----------------------------

    def _dynamic_top_k(self, query: str, requested_top_k: int) -> int:
        # simple heuristic: longer queries → more results
        length = len(query.split())
        if length <= 3:
            return min(requested_top_k, 5)
        elif length <= 10:
            return min(requested_top_k + 5, 15)
        else:
            return min(requested_top_k + 10, 25)

    # -----------------------------
    # Fallback logic
    # -----------------------------

    def _fallback_search(
        self,
        query: str,
        top_k: int,
        product_id: str | None = None,
        semantic_filters: dict[str, Iterable[str]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fallback: use Qdrant's full-scan with filter and simple keyword scoring.
        Assumes we can scroll all points (for small/medium collections).
        For large collections, you'd implement a dedicated keyword index.
        """
        logger.warning("QdrantRetriever: using fallback keyword-based search")

        filter_ = self._build_filter(product_id=product_id, semantic_filters=semantic_filters)

        # scroll all points (or limited batches)
        points: list[dict[str, Any]] = []
        offset = None

        while True:
            page = self.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=256,
                offset=offset,
                with_payload=True,
                filter=filter_,
            )
            batch_points, next_offset = page
            if not batch_points:
                break

            for p in batch_points:
                payload = p.payload or {}
                points.append(
                    {
                        "product_id": payload.get("product_id", ""),
                        "rating": payload.get("rating"),
                        "review_title": payload.get("review_title", ""),
                        "review_text": payload.get("review_text", ""),
                        "title": payload.get("title", ""),
                        "categories": payload.get("categories", ""),
                        "description": payload.get("description", ""),
                        "score": float(p.score) if getattr(p, "score", None) is not None else 0.0,
                    }
                )

            offset = next_offset
            if offset is None:
                break

        # keyword scoring
        for r in points:
            doc_text = " ".join(
                [
                    r.get("review_title", "") or "",
                    r.get("review_text", "") or "",
                    r.get("title", "") or "",
                    r.get("description", "") or "",
                ]
            )
            r["keyword_score"] = self._keyword_score(doc_text, query)

        points_sorted = sorted(points, key=lambda x: x.get("keyword_score", 0.0), reverse=True)
        return points_sorted[:top_k]

    # -----------------------------
    # Main search (hybrid + rerank + fallback)
    # -----------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        product_id: str | None = None,
        semantic_filters: dict[str, Iterable[str]] | None = None,
        use_hybrid: bool = True,
        use_rerank: bool = True,
    ) -> list[dict[str, Any]]:
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("QdrantRetriever: top_k must be a positive integer")

        # dynamic top_k
        effective_top_k = self._dynamic_top_k(query, top_k)

        RETRIEVAL_REQUESTS_TOTAL.labels(source="qdrant").inc()

        with RETRIEVAL_LATENCY_SECONDS.labels(source="qdrant").time():
            with self.tracer.start_as_current_span("qdrant.search") as span:
                span.set_attribute("query", query)
                span.set_attribute("requested_top_k", top_k)
                span.set_attribute("effective_top_k", effective_top_k)
                span.set_attribute("collection_name", COLLECTION_NAME)
                span.set_attribute("has_product_filter", product_id is not None)
                span.set_attribute("has_semantic_filters", semantic_filters is not None)

                if product_id:
                    span.set_attribute("product_id", str(product_id))

                query_vector = self._embed_query(query)
                query_filter = self._build_filter(product_id=product_id, semantic_filters=semantic_filters)

                try:
                    results = self.client.query_points(
                        collection_name=COLLECTION_NAME,
                        query=query_vector,
                        query_filter=query_filter,
                        limit=effective_top_k,
                        with_payload=True,
                    )
                except Exception:
                    logger.error(
                        "Qdrant query failed",
                        extra={
                            "collection_name": COLLECTION_NAME,
                            "top_k": effective_top_k,
                            "product_id": product_id,
                            "semantic_filters": semantic_filters,
                        },
                        exc_info=True,
                    )
                    # fallback logic
                    fallback_results = self._fallback_search(
                        query=query,
                        top_k=top_k,
                        product_id=product_id,
                        semantic_filters=semantic_filters,
                    )
                    span.set_attribute("results_count", len(fallback_results))
                    span.set_attribute("used_fallback", True)
                    return fallback_results

                output: list[dict[str, Any]] = []

                for point in results.points:
                    payload = point.payload or {}

                    base = {
                        "product_id": payload.get("product_id", ""),
                        "rating": payload.get("rating"),
                        "review_title": payload.get("review_title", ""),
                        "review_text": payload.get("review_text", ""),
                        "title": payload.get("title", ""),
                        "categories": payload.get("categories", ""),
                        "description": payload.get("description", ""),
                        "score": float(point.score),
                    }

                    if use_hybrid:
                        doc_text = " ".join(
                            [
                                base.get("review_title", "") or "",
                                base.get("review_text", "") or "",
                                base.get("title", "") or "",
                                base.get("description", "") or "",
                            ]
                        )
                        kw_score = self._keyword_score(doc_text, query)
                        base["keyword_score"] = kw_score
                        base["hybrid_score"] = self._hybrid_score(base["score"], kw_score)

                    output.append(base)

                # sort by hybrid_score if enabled, else by score
                if use_hybrid:
                    output = sorted(output, key=lambda x: x.get("hybrid_score", x["score"]), reverse=True)
                else:
                    output = sorted(output, key=lambda x: x["score"], reverse=True)

                # rerank with CrossEncoder
                if use_rerank:
                    output = self._rerank(query, output, top_k)

                # final truncate to requested top_k
                output = output[:top_k]

                span.set_attribute("results_count", len(output))
                span.set_attribute("used_fallback", False)
                return output

    # -----------------------------
    # Batch retrieval
    # -----------------------------

    def search_batch(
        self,
        queries: list[str],
        top_k: int = 5,
        product_id: str | None = None,
        semantic_filters: dict[str, Iterable[str]] | None = None,
        use_hybrid: bool = True,
        use_rerank: bool = True,
    ) -> list[list[dict[str, Any]]]:
        """
        Batch retrieval: run search for multiple queries.
        Returns a list of result lists, one per query.
        """
        results_batch: list[list[dict[str, Any]]] = []

        for q in queries:
            res = self.search(
                query=q,
                top_k=top_k,
                product_id=product_id,
                semantic_filters=semantic_filters,
                use_hybrid=use_hybrid,
                use_rerank=use_rerank,
            )
            results_batch.append(res)

        return results_batch
