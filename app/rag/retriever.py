from __future__ import annotations

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from app.logging.logger import get_logger
from app.observability.tracing import get_tracer
from app.config.paths import IMAGE_FAISS_INDEX_PATH, METADATA_PATH, EMBEDDINGS_PATH
logger = get_logger("rag.review_retriever")

_MODEL = None  # Singleton model


class ReviewRetriever:
    def __init__(
        self,
        index_path: str = IMAGE_FAISS_INDEX_PATH,
        metadata_path: str = METADATA_PATH,
        embeddings_path: str = EMBEDDINGS_PATH,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:

        self.index = faiss.read_index(index_path)
        self.metadata = pd.read_csv(metadata_path)
        self.embeddings = np.load(embeddings_path).astype("float32")

        if self.embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array [num_vectors, dim]")

        if np.isnan(self.embeddings).any():
            raise ValueError("Embeddings contain NaN values")

        if self.index.ntotal != len(self.metadata):
            raise ValueError("FAISS index size does not match metadata row count")

        global _MODEL
        if _MODEL is None:
            _MODEL = SentenceTransformer(model_name)

        self.model = _MODEL
        self.tracer = get_tracer("app.review_retriever")

    def embed_query(self, query: str) -> np.ndarray:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("ReviewRetriever: query must be a non-empty string")

        vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        return vector

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("ReviewRetriever: top_k must be a positive integer")

        with self.tracer.start_as_current_span("faiss.search"):
            query_vector = self.embed_query(query)

            try:
                scores, indices = self.index.search(query_vector, top_k)
            except Exception:
                logger.error("FAISS search failed", exc_info=True)
                raise

            results = self.metadata.iloc[indices[0]].copy()
            results["score"] = scores[0]

            return results.reset_index(drop=True)

    def search_by_product(self, product_id: str, query: str, top_k: int = 5) -> pd.DataFrame:
        if not isinstance(product_id, str):
            raise ValueError("ReviewRetriever: product_id must be a string")

        query_vector = self.embed_query(query)

        product_mask = self.metadata["product_id"].astype(str) == str(product_id)
        filtered_metadata = self.metadata[product_mask].copy()

        if filtered_metadata.empty:
            raise ValueError(f"No documents found for product_id={product_id}")

        filtered_indices = filtered_metadata.index.to_numpy()
        filtered_embeddings = self.embeddings[filtered_indices]

        similarities = np.dot(filtered_embeddings, query_vector[0])

        top_local_idx = np.argsort(similarities)[::-1][:top_k]
        top_global_idx = filtered_indices[top_local_idx]
        top_scores = similarities[top_local_idx]

        results = self.metadata.iloc[top_global_idx].copy()
        results["score"] = top_scores

        return results.reset_index(drop=True)
