from __future__ import annotations

import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from app.rag.qdrant_client_manager import get_qdrant_client


QDRANT_STORAGE_PATH = "artifacts/qdrant_storage"
COLLECTION_NAME = "review_embeddings"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class QdrantRetriever:
    def __init__(self) -> None:
        self.client = get_qdrant_client()
        self.model = SentenceTransformer(EMBED_MODEL_NAME)

    def close(self) -> None:
        self.client.close()

    def _embed_query(self, query: str) -> list[float]:
        vector = self.model.encode([query], normalize_embeddings=True)[0]
        return vector.astype(np.float32).tolist()

    def search(
        self,
        query: str,
        top_k: int = 5,
        product_id: str | None = None,
    ) -> list[dict]:
        query_vector = self._embed_query(query)

        query_filter = None
        if product_id is not None:
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

        return output


if __name__ == "__main__":
    retriever = QdrantRetriever()
    try:
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
    finally:
        retriever.close()