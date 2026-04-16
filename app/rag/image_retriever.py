from __future__ import annotations

import faiss
import numpy as np
import pandas as pd


class ImageRetriever:
    def __init__(
        self,
        index_path: str = "artifacts/indexes/image_faiss.index",
        metadata_path: str = "artifacts/embeddings/image_embedding_metadata.csv",
    ) -> None:
        self.index = faiss.read_index(index_path)
        self.metadata = pd.read_csv(metadata_path)

    def search_by_product(self, product_id: str, top_k: int = 5) -> pd.DataFrame:
        matches = self.metadata[self.metadata["product_id"].astype(str) == str(product_id)]
        if matches.empty:
            raise ValueError(f"Product not found in image metadata: {product_id}")

        query_idx = matches.index[0]
        query_vector = self.index.reconstruct(int(query_idx)).reshape(1, -1)

        scores, indices = self.index.search(query_vector, top_k + 1)

        top_scores = scores[0]
        top_indices = indices[0]

        results = []
        for idx, score in zip(top_indices, top_scores):
            row = self.metadata.iloc[idx]
            if str(row["product_id"]) == str(product_id):
                continue

            results.append(
                {
                    "product_id": row["product_id"],
                    "title": row.get("title", ""),
                    "image_url": row.get("image_url", ""),
                    "image_path": row.get("image_path", ""),
                    "similarity_score": float(score),
                }
            )

            if len(results) >= top_k:
                break

        return pd.DataFrame(results)


if __name__ == "__main__":
    retriever = ImageRetriever()
    results = retriever.search_by_product(product_id="B09SPZPDJK", top_k=5)

    print("\n=== VISUALLY SIMILAR PRODUCTS ===\n")
    print(results)