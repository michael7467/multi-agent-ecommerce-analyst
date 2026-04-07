from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


class ReviewsLoader:
    """
    Load local review data from JSONL/CSV/Parquet and normalize columns.
    """

    COLUMN_CANDIDATES = {
        "product_id": ["parent_asin", "asin"],
        "asin": ["asin"],
        "parent_asin": ["parent_asin"],
        "review_text": ["text", "reviewText", "review_text"],
        "rating": ["rating", "overall", "score"],
        "review_timestamp": ["timestamp", "unixReviewTime", "reviewTime"],
        "verified_purchase": ["verified_purchase", "verified", "verifiedPurchase"],
        "user_id": ["user_id", "reviewerID", "reviewer_id"],
        "review_title": ["title"],
    }

    REQUIRED_COLUMNS = ["product_id", "review_text", "rating"]

    OUTPUT_COLUMNS = [
        "product_id",
        "asin",
        "parent_asin",
        "review_text",
        "rating",
        "review_timestamp",
        "verified_purchase",
        "user_id",
        "review_title",
    ]

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def load(self, nrows: Optional[int] = None) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Reviews file not found: {self.file_path}")

        df = self._read_file(nrows=nrows)
        df = self._standardize_columns(df)
        df = self._validate_required_columns(df)
        df = self._keep_output_columns(df)
        return df

    def _read_file(self, nrows: Optional[int] = None) -> pd.DataFrame:
        name = self.file_path.name.lower()

        if name.endswith(".csv"):
            return pd.read_csv(self.file_path, nrows=nrows)

        if name.endswith(".jsonl"):
            return pd.read_json(self.file_path, lines=True, nrows=nrows)

        if name.endswith(".json"):
            return pd.read_json(self.file_path)

        if name.endswith(".parquet"):
            df = pd.read_parquet(self.file_path)
            return df.head(nrows) if nrows is not None else df

        raise ValueError(f"Unsupported file format: {self.file_path}")

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map: dict[str, str] = {}

        for target_col, candidates in self.COLUMN_CANDIDATES.items():
            for candidate in candidates:
                if candidate in df.columns:
                    rename_map[candidate] = target_col
                    break

        df = df.rename(columns=rename_map)

        if "product_id" not in df.columns:
            if "parent_asin" in df.columns:
                df["product_id"] = df["parent_asin"]
            elif "asin" in df.columns:
                df["product_id"] = df["asin"]

        return df

    def _validate_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required review columns: {missing}")
        return df

    def _keep_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [col for col in self.OUTPUT_COLUMNS if col in df.columns]
        return df[available].copy()


if __name__ == "__main__":
    loader = ReviewsLoader("data/raw/reviews_electronics_sample.jsonl")
    df = loader.load(nrows=5)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())