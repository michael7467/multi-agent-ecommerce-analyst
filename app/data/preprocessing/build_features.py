from __future__ import annotations

from pathlib import Path

import pandas as pd


class FeatureBuilder:
    def __init__(self, merged_df: pd.DataFrame) -> None:
        self.merged_df = merged_df.copy()

    def build(self) -> pd.DataFrame:
        df = self.merged_df.copy()

        # Review text length
        df["review_length"] = df["review_text"].fillna("").astype(str).str.len()

        # Ensure numeric columns are numeric
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df["verified_purchase"] = df["verified_purchase"].fillna(False).astype(int)
        df["review_timestamp"] = pd.to_numeric(df["review_timestamp"], errors="coerce")

        # Aggregate review-level data into product-level features
        agg_df = (
            df.groupby("product_id")
            .agg(
                review_count=("review_text", "count"),
                avg_rating=("rating", "mean"),
                rating_std=("rating", "std"),
                verified_purchase_ratio=("verified_purchase", "mean"),
                avg_review_length=("review_length", "mean"),
                min_review_timestamp=("review_timestamp", "min"),
                max_review_timestamp=("review_timestamp", "max"),
                title=("title", "first"),
                price=("price", "first"),
                categories=("categories", "first"),
                description=("description", "first"),
                store=("store", "first"),
            )
            .reset_index()
        )

        # Fill missing values
        agg_df["rating_std"] = agg_df["rating_std"].fillna(0.0)
        agg_df["price"] = pd.to_numeric(agg_df["price"], errors="coerce")

        text_cols = ["title", "categories", "description", "store"]
        for col in text_cols:
            agg_df[col] = agg_df[col].fillna("")

        # Optional derived feature: review time span
        agg_df["review_time_span"] = (
            agg_df["max_review_timestamp"] - agg_df["min_review_timestamp"]
        )

        return agg_df


def save_product_features(
    input_path: str = "data/processed/electronics_merged.csv",
    output_path: str = "data/processed/electronics_features.csv",
) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    builder = FeatureBuilder(df)
    features_df = builder.build()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)

    print(f"Saved features to: {output_path}")
    print(f"Feature table shape: {features_df.shape}")

    print("\nColumns:")
    print(features_df.columns.tolist())

    print("\nSample:")
    print(features_df.head())

    return features_df


if __name__ == "__main__":
    save_product_features()