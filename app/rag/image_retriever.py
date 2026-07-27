import faiss
import numpy as np
import pandas as pd

from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent
from app.config.paths import IMAGE_FAISS_INDEX_PATH, IMAGE_METADATA_PATH

logger = get_logger("retriever.image")


class ImageRetriever:
    def __init__(
        self,
        index_path: str = str(IMAGE_FAISS_INDEX_PATH),
        metadata_path: str = str(IMAGE_METADATA_PATH),
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

        # search_by_product below uses a row's index LABEL directly as its
        # POSITION in the FAISS index (via index.reconstruct()). A fresh
        # read_csv() already gives a clean 0..n-1 range today, so this
        # changes nothing right now -- it's here so a future filter added
        # above this line (e.g. dropping rows with missing images) can't
        # silently break that alignment. Same fix as RecommenderService
        # last turn, more consequential here since it feeds reconstruct()
        # directly rather than a matrix lookup.
        self.metadata = self.metadata.reset_index(drop=True)

        # index.reconstruct() only works directly on flat indexes.
        # IndexIVF-family indexes raise "direct map is not initialized"
        # unless make_direct_map() has been called first (confirmed
        # against FAISS's own docs/issue tracker, not assumed) -- and nothing
        # in this codebase calls it anywhere. hasattr guards this so it's a
        # no-op for flat indexes (which don't have this method at all) and
        # a real fix for IVF-family ones, without needing to know which
        # this actually is.
        if hasattr(self.index, "make_direct_map"):
            self.index.make_direct_map()

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

        scores, indices = self.index.search(query_vector, top_k + 1)

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