from __future__ import annotations

import faiss
import numpy as np
import pandas as pd
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent
from app.config.paths import INDEX_PATH, METADATA_PATH

logger = get_logger("retriever.image")

class ImageRetriever:
    def __init__(
        self,
        index_path: str = INDEX_PATH,
        metadata_path: str = METADATA_PATH,
    ) -> None:
        try:
            self.index = faiss.read_index(index_path)
        except Exception:
            logger.error("Failed to load FAISS index", exc_info=True)
            raise

        try:
            self.metadata = pd.read_csv(metadata_path)
        except Exception:
            logger.error("Failed to load image metadata CSV", exc_info=True)
            raise

        required = ["product_id", "image_url"]
        for col in required:
            if col not in self.metadata.columns:
                raise ValueError(f"Missing metadata column: {col}")

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"FAISS index size ({self.index.ntotal}) does not match metadata rows ({len(self.metadata)})"
            )

    @traced_agent("image_retriever_search")
    def search_by_product(self, product_id: str, top_k: int = 5) -> pd.DataFrame:
        if not isinstance(product_id, str):
            raise ValueError("ImageRetriever: product_id must be a string")

        matches = self.metadata[self.metadata["product_id"].astype(str) == str(product_id)]
        if matches.empty:
            raise ValueError(f"Product not found in image metadata: {product_id}")

        query_idx = matches.index[0]
        query_vector = self.index.reconstruct(int(query_idx)).reshape(1, -1)

        # Normalize for cosine similarity
        faiss.normalize_L2(query_vector)

        try:
            scores, indices = self.index.search(query_vector, top_k + 1)
        except Exception:
            logger.error("FAISS search failed", exc_info=True)
            raise

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue

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
