from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.data.loaders.metadata_loader import MetadataLoader


class MetadataCleaner:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    def clean(self) -> pd.DataFrame:
        df = self.df.copy()

        # Drop rows missing product_id
        df = df.dropna(subset=["product_id"])

        # Normalize text columns
        text_cols = ["title", "store", "brand"]
        for col in text_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .fillna("")
                    .astype(str)
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip()
                )

        # Normalize price
        if "price" in df.columns:
            df["price"] = (
                df["price"]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df["price"] = pd.to_numeric(df["price"], errors="coerce")

        # Convert list-like columns to string for easier CSV storage
        for col in ["categories", "description"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
                df[col] = df[col].apply(lambda x: " | ".join(map(str, x)))

        # Remove duplicates by product_id
        df = df.drop_duplicates(subset=["product_id"])

        # Reset index
        df = df.reset_index(drop=True)

        return df


def save_clean_metadata(
    input_path: str = "data/raw/meta_electronics_sample.jsonl",
    output_path: str = "data/interim/meta_electronics_clean.csv",
    nrows: int | None = None,
) -> pd.DataFrame:
    loader = MetadataLoader(input_path)
    df = loader.load(nrows=nrows)

    cleaner = MetadataCleaner(df)
    clean_df = cleaner.clean()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_path, index=False)

    print(f"Saved cleaned metadata to: {output_path}")
    print(f"Shape: {clean_df.shape}")
    print(clean_df.head())

    return clean_df


if __name__ == "__main__":
    save_clean_metadata()