from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataMerger:
    def __init__(self, reviews_df: pd.DataFrame, metadata_df: pd.DataFrame) -> None:
        self.reviews_df = reviews_df.copy()
        self.metadata_df = metadata_df.copy()

    def merge(self) -> pd.DataFrame:
        reviews = self.reviews_df.copy()
        metadata = self.metadata_df.copy()

        merged_df = reviews.merge(
            metadata,
            on="product_id",
            how="left",
            suffixes=("_review", "_meta"),
        )

        return merged_df


def save_merged_data(
    reviews_path: str = "data/interim/reviews_electronics_clean.csv",
    metadata_path: str = "data/interim/meta_electronics_clean.csv",
    output_path: str = "data/processed/electronics_merged.csv",
) -> pd.DataFrame:
    reviews_df = pd.read_csv(reviews_path)
    metadata_df = pd.read_csv(metadata_path)

    merger = DataMerger(reviews_df, metadata_df)
    merged_df = merger.merge()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    print(f"Saved merged data to: {output_path}")
    print(f"Merged shape: {merged_df.shape}")

    matched_rows = merged_df["title"].notna().sum() if "title" in merged_df.columns else 0
    print(f"Rows with matched metadata: {matched_rows}/{len(merged_df)}")

    print("\nColumns:")
    print(merged_df.columns.tolist())

    print("\nSample:")
    print(merged_df.head())

    return merged_df


if __name__ == "__main__":
    save_merged_data()