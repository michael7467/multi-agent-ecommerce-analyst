from __future__ import annotations

import numpy as np
from qdrant_client import models
from sentence_transformers import SentenceTransformer

from app.observability.metrics import RETRIEVAL_LATENCY_SECONDS, RETRIEVAL_REQUESTS_TOTAL
from app.observability.tracing import get_tracer
from app.rag.qdrant_client_manager import get_qdrant_client
from app.core.config import settings

COLLECTION_NAME = settings.qdrant_collection_name
EMBED_MODEL_NAME = settings.embedding_model_name


class QdrantRetriever:
    def __init__(self) -> None:
        self.client = get_qdrant_client()
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        self.tracer = get_tracer("app.qdrant_retriever")

    def _embed_query(self, query: str) -> list[float]:
        vector = self.model.encode([query], normalize_embeddings=True)[0]
        return vector.astype(np.float32).tolist()

    def search(
        self,
        query: str,
        top_k: int = 5,
        product_id: str | None = None,
    ) -> list[dict]:
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

                results = self.client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                )

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


if __name__ == "__main__":
    retriever = QdrantRetriever()

    print("\n=== GLOBAL SEARCH ===")
    global_results = retriever.search(
        query="sound quality and noise cancellation",
        top_k=5,
    )
    for r in global_results:
        print(r)

    print("\n=== PRODUCT FILTERED SEARCH ===")
    filtered_results = retriever.search(
        query="sound quality and noise cancellation",
        top_k=3,
        product_id="B09SPZPDJK",
    )
    for r in filtered_results:
        print(r)