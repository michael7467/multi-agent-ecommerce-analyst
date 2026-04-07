from __future__ import annotations

from pathlib import Path

import pandas as pd


class LabelCreator:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    def create_price_class_labels(self) -> pd.DataFrame:
        df = self.df.copy()

        # Ensure price is numeric
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

        # Remove missing or non-positive prices
        df = df.dropna(subset=["price"])
        df = df[df["price"] > 0].copy()

        # Create 3 price classes using quantiles
        df["price_class"] = pd.qcut(
            df["price"],
            q=3,
            labels=["low", "mid", "high"],
            duplicates="drop",
        )

        return df.reset_index(drop=True)


def save_labeled_data(
    input_path: str = "data/processed/electronics_features.csv",
    output_path: str = "data/processed/electronics_labeled.csv",
) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    creator = LabelCreator(df)
    labeled_df = creator.create_price_class_labels()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(output_path, index=False)

    print(f"Saved labeled data to: {output_path}")
    print(f"Shape: {labeled_df.shape}")

    print("\nPrice class distribution:")
    print(labeled_df["price_class"].value_counts())

    print("\nSample:")
    print(labeled_df[["product_id", "price", "price_class"]].head())

    return labeled_df


if __name__ == "__main__":
    save_labeled_data()