from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class ReviewEmbedder:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings


def save_review_embeddings(
    input_path: str = "data/processed/review_documents.csv",
    embeddings_output_path: str = "artifacts/embeddings/review_embeddings.npy",
    metadata_output_path: str = "artifacts/embeddings/review_embedding_metadata.csv",
) -> None:
    df = pd.read_csv(input_path)

    if "document_text" not in df.columns:
        raise ValueError("Missing required column: document_text")

    df["document_text"] = df["document_text"].fillna("").astype(str)

    embedder = ReviewEmbedder()
    embeddings = embedder.embed_documents(df["document_text"].tolist())

    Path(embeddings_output_path).parent.mkdir(parents=True, exist_ok=True)

    np.save(embeddings_output_path, embeddings)
    df.to_csv(metadata_output_path, index=False)

    print(f"Saved embeddings to: {embeddings_output_path}")
    print(f"Saved embedding metadata to: {metadata_output_path}")
    print(f"Embeddings shape: {embeddings.shape}")
    print("\nMetadata sample:")
    print(df.head())


if __name__ == "__main__":
    save_review_embeddings()