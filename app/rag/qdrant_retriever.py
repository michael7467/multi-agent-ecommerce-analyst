from __future__ import annotations

from typing import Any

import numpy as np
from qdrant_client import models
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding

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

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RERANK_CANDIDATE_MULTIPLIER = 4


DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"

_MODEL: SentenceTransformer | None = None
_SPARSE_MODEL: SparseTextEmbedding | None = None
_RERANK_MODEL: CrossEncoder | None = None


class QdrantRetriever:
    def __init__(self) -> None:
        global _MODEL, _SPARSE_MODEL, _RERANK_MODEL

        self.client = get_qdrant_client()
        self.tracer = get_tracer("app.qdrant_retriever")

        if _MODEL is None:
            logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
            _MODEL = SentenceTransformer(EMBED_MODEL_NAME)

        if _SPARSE_MODEL is None:
            logger.info("Loading sparse BM25 model: Qdrant/bm25")

            _SPARSE_MODEL = SparseTextEmbedding("Qdrant/bm25")

        if _RERANK_MODEL is None:
            logger.info(f"Loading reranking cross-encoder: {RERANK_MODEL_NAME}")
            _RERANK_MODEL = CrossEncoder(RERANK_MODEL_NAME)

        self.model = _MODEL
        self.sparse_model = _SPARSE_MODEL
        self.rerank_model = _RERANK_MODEL

        if not self.client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Qdrant collection '{COLLECTION_NAME}' does not exist. "
                "Build/upload the Qdrant index before running retrieval."
            )

    def _embed_query(self, query: str) -> list[float]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("QdrantRetriever: query must be a non-empty string")

        vector = self.model.encode(
            [query.strip()],
            normalize_embeddings=True,
        )[0]

        return vector.astype(np.float32).tolist()

    def _embed_sparse_query(self, query: str) -> models.SparseVector:
        sparse = list(self.sparse_model.embed([query.strip()]))[0]
        return models.SparseVector(
            indices=sparse.indices.tolist(),
            values=sparse.values.tolist(),
        )

    def _build_filter(self, product_id: str | None) -> models.Filter | None:
        if not product_id:
            return None

        return models.Filter(
            must=[
                models.FieldCondition(
                    key="product_id",
                    match=models.MatchValue(value=str(product_id)),
                )
            ]
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        product_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("QdrantRetriever: top_k must be a positive integer")

        RETRIEVAL_REQUESTS_TOTAL.labels(source="qdrant").inc()

        with RETRIEVAL_LATENCY_SECONDS.labels(source="qdrant").time():
            with self.tracer.start_as_current_span("qdrant.search") as span:
                span.set_attribute("query", query)
                span.set_attribute("top_k", top_k)
                span.set_attribute("collection_name", COLLECTION_NAME)
                span.set_attribute("has_product_filter", product_id is not None)
                span.set_attribute("hybrid", True)

                if product_id:
                    span.set_attribute("product_id", str(product_id))

                dense_vector = self._embed_query(query)
                sparse_vector = self._embed_sparse_query(query)
     
                query_filter = self._build_filter(product_id)

                try:
          
                    results = self.client.query_points(
                        collection_name=COLLECTION_NAME,
                        prefetch=[
                            models.Prefetch(
                                query=dense_vector,
                                using=DENSE_VECTOR_NAME,
                                filter=query_filter,
                                limit=top_k * 2,
                            ),
                            models.Prefetch(
                                query=sparse_vector,
                                using=SPARSE_VECTOR_NAME,
                                filter=query_filter,
                                limit=top_k * 2,
                            ),
                        ],
                        query=models.FusionQuery(fusion=models.Fusion.RRF),
          
                        limit=top_k * RERANK_CANDIDATE_MULTIPLIER,
                        with_payload=True,
                    )
                except Exception:
                    logger.error(
                        "Qdrant hybrid query failed",
                        extra={
                            "collection_name": COLLECTION_NAME,
                            "top_k": top_k,
                            "product_id": product_id,
                        },
                        exc_info=True,
                    )
                    raise

                output: list[dict[str, Any]] = []

                for point in results.points:
                    payload = point.payload or {}

                    output.append(
                        {
                            "product_id": payload.get("product_id", ""),
                            "rating": payload.get("rating"),
                            "review_title": payload.get("review_title", ""),
                            "review_text": payload.get("review_text", ""),
                            "title": payload.get("title", ""),
                            "categories": payload.get("categories", ""),
                            "description": payload.get("description", ""),
               
                            "score": float(point.score),
                        }
                    )

    
                if output:
                    pairs = [
                        (query, f"{item['review_title']} {item['review_text']}".strip())
                        for item in output
                    ]
                    rerank_scores = self.rerank_model.predict(pairs)

                    for item, rerank_score in zip(output, rerank_scores):
                        item["score"] = float(rerank_score)

                    output.sort(key=lambda item: item["score"], reverse=True)
                    output = output[:top_k]

                span.set_attribute("results_count", len(output))
                return output