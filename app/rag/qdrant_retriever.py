from __future__ import annotations

import numpy as np
from qdrant_client import models
from sentence_transformers import SentenceTransformer

from app.observability.metrics import RETRIEVAL_LATENCY_SECONDS, RETRIEVAL_REQUESTS_TOTAL
from app.observability.tracing import get_tracer
from app.rag.qdrant_client_manager import get_qdrant_client
from app.core.config import settings
from app.logging.logger import get_logger

logger = get_logger("rag.qdrant")

COLLECTION_NAME = settings.qdrant_collection_name
EMBED_MODEL_NAME = settings.embedding_model_name

# Singleton model to avoid repeated loading
_MODEL = None


class QdrantRetriever:
    def __init__(self) -> None:
        global _MODEL

        self.client = get_qdrant_client()

        if _MODEL is None:
            _MODEL = SentenceTransformer(EMBED_MODEL_NAME)

        self.model = _MODEL
        self.tracer = get_tracer("app.qdrant_retriever")

        if not self.client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(f"Qdrant collection '{COLLECTION_NAME}' does not exist")

    def _embed_query(self, query: str) -> list[float]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("QdrantRetriever: query must be a non-empty string")

        vector = self.model.encode([query], normalize_embeddings=True)[0]
        return vector.astype(np.float32).tolist()

    def search(
        self,
        query: str,
        top_k: int = 5,
        product_id: str | None = None,
    ) -> list[dict]:
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("QdrantRetriever: top_k must be a positive integer")

        RETRIEVAL_REQUESTS_TOTAL.inc()

        with RETRIEVAL_LATENCY_SECONDS.time():
            with self.tracer.start_as_current_span("qdrant.search") as span:
                span.set_attribute("query", query)
                span.set_attribute("top_k", int(top_k))
                span.set_attribute("has_product_filter", product_id is not None)

                query_vector = self._embed_query(query)

                query_filter = None
                if product_id is not None:
                    span.set_attribute("product_id", str(product_id))
                    query_filter = models.Filter(
                        must=[
                            models.FieldCondition(
                                key="product_id",
                                match=models.MatchValue(value=str(product_id)),
                            )
                        ]
                    )

                try:
                    results = self.client.query_points(
                        collection_name=COLLECTION_NAME,
                        query=query_vector,
                        query_filter=query_filter,
                        limit=top_k,
                        with_payload=True,
                    )
                except Exception:
                    logger.error("Qdrant query failed", exc_info=True)
                    raise

                output = []
                for point in results.points:
                    payload = point.payload or {}
                    output.append(
                        {
                            "product_id": payload.get("product_id", ""),
                            "rating": payload.get("rating", None),
                            "review_title": payload.get("review_title", ""),
                            "review_text": payload.get("review_text", ""),
                            "title": payload.get("title", ""),
                            "categories": payload.get("categories", ""),
                            "description": payload.get("description", ""),
                            "score": float(point.score),
                        }
                    )

                span.set_attribute("results_count", len(output))
                return output
