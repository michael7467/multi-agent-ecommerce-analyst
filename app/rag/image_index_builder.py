from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd


class ImageFaissIndexBuilder:
    def __init__(self, embeddings: np.ndarray) -> None:
        self.embeddings = embeddings.astype("float32")

    def build_index(self) -> faiss.Index:
        embedding_dim = self.embeddings.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(self.embeddings)
        return index


def save_image_faiss_index(
    embeddings_path: str = "artifacts/embeddings/image_embeddings.npy",
    metadata_path: str = "artifacts/embeddings/image_embedding_metadata.csv",
    index_output_path: str = "artifacts/indexes/image_faiss.index",
) -> None:
    embeddings = np.load(embeddings_path)
    metadata_df = pd.read_csv(metadata_path)

    if len(embeddings) != len(metadata_df):
        raise ValueError(
            f"Embeddings count ({len(embeddings)}) does not match metadata count ({len(metadata_df)})"
        )

    builder = ImageFaissIndexBuilder(embeddings)
    index = builder.build_index()

    Path(index_output_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_output_path)

    print(f"Saved image FAISS index to: {index_output_path}")
    print(f"Total vectors indexed: {index.ntotal}")
    print(f"Embedding dimension: {embeddings.shape[1]}")


if __name__ == "__main__":
    save_image_faiss_index()