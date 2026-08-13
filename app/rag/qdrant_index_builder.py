from __future__ import annotations

import numpy as np
import pandas as pd
from qdrant_client import models
from fastembed import SparseTextEmbedding

from app.rag.qdrant_client_manager import get_qdrant_client
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent
from app.config.paths import EMBEDDINGS_PATH, METADATA_PATH
from app.config.settings import settings

logger = get_logger("qdrant.index_builder")

COLLECTION_NAME = settings.qdrant_collection_name
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"


class QdrantIndexBuilder:
    def __init__(self) -> None:
        self.client = get_qdrant_client()
        self.sparse_model = SparseTextEmbedding("Qdrant/bm25")

    def create_or_replace_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(COLLECTION_NAME):
            self.client.delete_collection(COLLECTION_NAME)

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(),
            },
        )

    def upload_points(self, embeddings: np.ndarray, metadata_df: pd.DataFrame, batch_size: int = 256) -> None:
        payloads = metadata_df.to_dict(orient="records")
        ids = list(range(len(metadata_df)))

        texts = metadata_df["document_text"].tolist()
        sparse_vectors = list(self.sparse_model.embed(texts))

        points = [
            models.PointStruct(
                id=ids[i],
                vector={
                    DENSE_VECTOR_NAME: embeddings[i].tolist(),
                    SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=sparse_vectors[i].indices.tolist(),
                        values=sparse_vectors[i].values.tolist(),
                    ),
                },
                payload=payloads[i],
            )
            for i in range(len(metadata_df))
        ]

        total_batches = (len(points) + batch_size - 1) // batch_size
        for batch_num, start in enumerate(range(0, len(points), batch_size), start=1):
            batch = points[start : start + batch_size]
            self.client.upsert(collection_name=COLLECTION_NAME, points=batch)
            logger.info(f"Uploaded batch {batch_num}/{total_batches} ({len(batch)} points)")

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

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        vector_size = embeddings.shape[1]
        self.create_or_replace_collection(vector_size=vector_size)
        self.upload_points(embeddings=embeddings, metadata_df=metadata_df)

        logger.info(
            f"Built Qdrant collection '{COLLECTION_NAME}' "
            f"with {len(metadata_df)} points (dim={vector_size}, hybrid dense+bm25)"
        )

    def close(self) -> None:
        self.client.close()


if __name__ == "__main__":
    builder = QdrantIndexBuilder()
    try:
        builder.build()
    finally:
        builder.close()
