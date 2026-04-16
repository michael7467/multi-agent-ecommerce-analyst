from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models
from app.rag.qdrant_client_manager import get_qdrant_client

EMBEDDINGS_PATH = Path("artifacts/embeddings/review_embeddings.npy")
METADATA_PATH = Path("artifacts/embeddings/review_embedding_metadata.csv")
QDRANT_STORAGE_PATH = Path("artifacts/qdrant_storage")
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
        points = []

        for idx, (_, row) in enumerate(metadata_df.iterrows()):
            payload = {
                "product_id": str(row.get("product_id", "")),
                "rating": float(row.get("rating", 0.0)) if pd.notna(row.get("rating")) else None,
                "review_title": str(row.get("review_title", "")),
                "review_text": str(row.get("review_text", "")),
                "title": str(row.get("title", "")),
                "categories": str(row.get("categories", "")),
                "description": str(row.get("description", "")),
            }

            points.append(
                models.PointStruct(
                    id=idx,
                    vector=embeddings[idx].tolist(),
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

    def build(self) -> None:
        if not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(f"Missing embeddings file: {EMBEDDINGS_PATH}")
        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH}")

        embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
        metadata_df = pd.read_csv(METADATA_PATH)

        if len(embeddings) != len(metadata_df):
            raise ValueError(
                f"Embeddings count ({len(embeddings)}) does not match metadata count ({len(metadata_df)})"
            )

        vector_size = embeddings.shape[1]
        self.create_or_replace_collection(vector_size=vector_size)
        self.upload_points(embeddings=embeddings, metadata_df=metadata_df)

        print(f"Built Qdrant collection: {COLLECTION_NAME}")
        print(f"Stored points: {len(metadata_df)}")
        print(f"Vector size: {vector_size}")
        print(f"Storage path: {QDRANT_STORAGE_PATH}")

    def close(self) -> None:
        self.client.close()


if __name__ == "__main__":
    builder = QdrantIndexBuilder()
    try:
        builder.build()
    finally:
        builder.close()