from __future__ import annotations

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class ReviewRetriever:
    def __init__(
        self,
        index_path: str = "artifacts/indexes/review_faiss.index",
        metadata_path: str = "artifacts/embeddings/review_embedding_metadata.csv",
        embeddings_path: str = "artifacts/embeddings/review_embeddings.npy",
        model_name: str = MODEL_NAME,
    ) -> None:
        self.index = faiss.read_index(index_path)
        self.metadata = pd.read_csv(metadata_path)
        self.embeddings = np.load(embeddings_path).astype("float32")
        self.model = SentenceTransformer(model_name)

    def embed_query(self, query: str) -> np.ndarray:
        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        return query_vector

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        query_vector = self.embed_query(query)

        scores, indices = self.index.search(query_vector, top_k)
        top_scores = scores[0]
        top_indices = indices[0]

        results = self.metadata.iloc[top_indices].copy()
        results["score"] = top_scores

        return results.reset_index(drop=True)

    def search_by_product(self, product_id: str, query: str, top_k: int = 5) -> pd.DataFrame:
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


if __name__ == "__main__":
    retriever = ReviewRetriever()

    print("\n=== Global Retrieval ===")
    global_query = "premium wireless noise cancelling headphones with great sound quality"
    global_results = retriever.search(query=global_query, top_k=5)
    print(global_results[["product_id", "title", "categories", "review_text", "score"]])

    print("\n=== Product-Specific Retrieval ===")
    sample_product_id = global_results.iloc[0]["product_id"]
    product_query = "sound quality and noise cancellation"
    product_results = retriever.search_by_product(
        product_id=sample_product_id,
        query=product_query,
        top_k=3,
    )
    print(product_results[["product_id", "title", "review_text", "score"]])