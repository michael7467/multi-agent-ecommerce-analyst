from __future__ import annotations

from pathlib import Path
import pandas as pd
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent

logger = get_logger("rag.review_chunk_builder")

class ReviewChunkBuilder:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

        required = ["product_id", "review_text"]
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"ReviewChunkBuilder: missing required column '{col}'")

    @traced_agent("ReviewChunkBuilder.build_documents")
    def build_documents(self) -> pd.DataFrame:
        df = self.df.copy()

        # Normalize text fields
        for col in ["review_title", "review_text", "title", "categories", "description"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str).str.strip()

        # Build unified retrieval text
        df["document_text"] = (
            "[PRODUCT]\n"
            "Title: " + df.get("title", "") + "\n"
            "Categories: " + df.get("categories", "") + "\n\n"
            "[REVIEW]\n"
            "Title: " + df.get("review_title", "") + "\n"
            "Text: " + df.get("review_text", "") + "\n\n"
            "[DESCRIPTION]\n"
            + df.get("description", "")
        )

        # Normalize whitespace
        df["document_text"] = df["document_text"].str.replace(r"\n+", "\n", regex=True)

        keep_cols = [
            "product_id",
            "rating",
            "review_title",
            "review_text",
            "title",
            "categories",
            "description",
            "document_text",
        ]

        available_cols = [col for col in keep_cols if col in df.columns]
        docs = df[available_cols].copy()

        # Deduplicate
        dedupe_cols = [col for col in ["product_id", "review_text"] if col in docs.columns]
        if dedupe_cols:
            docs = docs.drop_duplicates(subset=dedupe_cols)

        docs = docs.reset_index(drop=True)
        return docs


def save_review_documents(
    input_path: str = "data/processed/electronics_merged.csv",
    output_path: str = "data/processed/review_documents.csv",
) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    builder = ReviewChunkBuilder(df)
    docs_df = builder.build_documents()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    docs_df.to_csv(output_path, index=False)

    print(f"Saved review documents to: {output_path}")
    print(f"Shape: {docs_df.shape}")
    print("\nColumns:", docs_df.columns.tolist())
    print("\nSample:")
    print(docs_df.head())

    return docs_df
