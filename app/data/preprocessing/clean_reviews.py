from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.data.loaders.reviews_loader import ReviewsLoader


class ReviewCleaner:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    def clean(self) -> pd.DataFrame:
        df = self.df.copy()

        # Drop rows missing critical fields
        df = df.dropna(subset=["product_id", "review_text", "rating"])

        # Normalize text
        df["review_text"] = (
            df["review_text"]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        # Remove empty review text
        df = df[df["review_text"] != ""]

        # Normalize rating
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df.dropna(subset=["rating"])
        df = df[(df["rating"] >= 1) & (df["rating"] <= 5)]

        # Normalize verified_purchase if present
        if "verified_purchase" in df.columns:
            df["verified_purchase"] = df["verified_purchase"].fillna(False).astype(bool)

        # Convert timestamp if present
        if "review_timestamp" in df.columns:
            df["review_timestamp"] = pd.to_numeric(df["review_timestamp"], errors="coerce")
            df["review_datetime"] = pd.to_datetime(
                df["review_timestamp"], unit="ms", errors="coerce"
            )

            # fallback if timestamps are in seconds
            missing_dt = df["review_datetime"].isna()
            if missing_dt.any():
                df.loc[missing_dt, "review_datetime"] = pd.to_datetime(
                    df.loc[missing_dt, "review_timestamp"],
                    unit="s",
                    errors="coerce",
                )

        # Remove duplicates
        dedupe_cols = [c for c in ["user_id", "product_id", "review_text", "rating"] if c in df.columns]
        if dedupe_cols:
            df = df.drop_duplicates(subset=dedupe_cols)
        else:
            df = df.drop_duplicates()

        # Reset index
        df = df.reset_index(drop=True)

        return df


def save_clean_reviews(
    input_path: str = "data/raw/reviews_electronics_sample.jsonl",
    output_path: str = "data/interim/reviews_electronics_clean.csv",
    nrows: int | None = None,
) -> pd.DataFrame:
    loader = ReviewsLoader(input_path)
    df = loader.load(nrows=nrows)

    cleaner = ReviewCleaner(df)
    clean_df = cleaner.clean()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_path, index=False)

    print(f"Saved cleaned reviews to: {output_path}")
    print(f"Shape: {clean_df.shape}")
    print(clean_df.head())

    return clean_df


if __name__ == "__main__":
    save_clean_reviews()