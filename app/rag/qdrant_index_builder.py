from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models
from app.rag.qdrant_client_manager import get_qdrant_client
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent
from app.config.paths import EMBEDDINGS_PATH, METADATA_PATH

logger = get_logger("qdrant.index_builder")


COLLECTION_NAME = "review_embeddings"


class QdrantIndexBuilder:
    def __init__(self) -> None:
        self.client = get_qdrant_client()

    def create_or_replace_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(COLLECTION_NAME):
            self.client.delete_collection(COLLECTION_NAME)

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def upload_points(self, embeddings: np.ndarray, metadata_df: pd.DataFrame) -> None:
        payloads = metadata_df.to_dict(orient="records")
        ids = list(range(len(metadata_df)))

        self.client.upload_collection(
            collection_name=COLLECTION_NAME,
            vectors=embeddings,
            payload=payloads,
            ids=ids,
        )

    @traced_agent("qdrant_index_build")
    def build(self) -> None:
        if not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(f"Missing embeddings file: {EMBEDDINGS_PATH}")
        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH}")

        embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
        metadata_df = pd.read_csv(METADATA_PATH)

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array [num_vectors, dim]")

        if np.isnan(embeddings).any():
            raise ValueError("Embeddings contain NaN values")

        if len(embeddings) != len(metadata_df):
            raise ValueError(
                f"Embeddings count ({len(embeddings)}) does not match metadata count ({len(metadata_df)})"
            )

        # Normalize for cosine similarity
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        vector_size = embeddings.shape[1]
        self.create_or_replace_collection(vector_size=vector_size)
        self.upload_points(embeddings=embeddings, metadata_df=metadata_df)

        logger.info(
            f"Built Qdrant collection '{COLLECTION_NAME}' "
            f"with {len(metadata_df)} points (dim={vector_size})"
        )

    def close(self) -> None:
        self.client.close()

        
if __name__ == "__main__":
    builder = QdrantIndexBuilder()
    try:
        builder.build()
    finally:
        builder.close()