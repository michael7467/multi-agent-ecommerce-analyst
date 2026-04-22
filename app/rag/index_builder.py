from __future__ import annotations

from pathlib import Path
import faiss
import numpy as np
import pandas as pd

from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent

logger = get_logger("faiss.review_index")

class FaissIndexBuilder:
    def __init__(self, embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array [num_vectors, dim]")

        if np.isnan(embeddings).any():
            raise ValueError("Embeddings contain NaN values")

        self.embeddings = embeddings.astype("float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)

    @traced_agent("faiss_index_build")
    def build_index(self) -> faiss.Index:
        try:
            embedding_dim = self.embeddings.shape[1]
            index = faiss.IndexFlatIP(embedding_dim)
            index.add(self.embeddings)
            return index
        except Exception:
            logger.error("Failed to build FAISS index", exc_info=True)
            raise


@traced_agent("faiss_index_save")
def save_faiss_index(
    embeddings_path: str = "artifacts/embeddings/review_embeddings.npy",
    metadata_path: str = "artifacts/embeddings/review_embedding_metadata.csv",
    index_output_path: str = "artifacts/indexes/review_faiss.index",
) -> None:
    embeddings = np.load(embeddings_path)
    metadata_df = pd.read_csv(metadata_path)

    if len(embeddings) != len(metadata_df):
        raise ValueError(
            f"Embeddings count ({len(embeddings)}) does not match metadata count ({len(metadata_df)})"
        )

    required = ["product_id"]
    for col in required:
        if col not in metadata_df.columns:
            raise ValueError(f"Missing metadata column: {col}")

    builder = FaissIndexBuilder(embeddings)
    index = builder.build_index()

    if index.ntotal != len(embeddings):
        raise RuntimeError("FAISS index size mismatch after building")

    Path(index_output_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_output_path)

    logger.info(
        f"Saved review FAISS index to {index_output_path} "
        f"({index.ntotal} vectors, dim={embeddings.shape[1]})"
    )
