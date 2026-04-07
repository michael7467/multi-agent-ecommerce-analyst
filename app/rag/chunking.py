from __future__ import annotations

from pathlib import Path
import pandas as pd


class ReviewChunkBuilder:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    def build_documents(self) -> pd.DataFrame:
        df = self.df.copy()

        # Normalize text fields
        for col in ["review_title", "review_text", "title", "categories", "description"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str).str.strip()

        # Build one text field for retrieval
        df["document_text"] = (
            "Product Title: " + df.get("title", pd.Series("", index=df.index)) + "\n"
            + "Categories: " + df.get("categories", pd.Series("", index=df.index)) + "\n"
            + "Review Title: " + df.get("review_title", pd.Series("", index=df.index)) + "\n"
            + "Review Text: " + df.get("review_text", pd.Series("", index=df.index)) + "\n"
            + "Description: " + df.get("description", pd.Series("", index=df.index))
        )

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

        # Remove duplicates
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

    print("\nColumns:")
    print(docs_df.columns.tolist())

    print("\nSample:")
    print(docs_df.head())

    return docs_df


if __name__ == "__main__":
    save_review_documents()