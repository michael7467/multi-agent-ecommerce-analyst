from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests


CSV_PATH = Path("data/processed/electronics_image_urls.csv")
OUTPUT_DIR = Path("data/raw/product_images")


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def download_images(limit: int | None = None) -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing file: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    if limit is not None:
        df = df.head(limit)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    failed = 0

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0"
        }
    )

    for _, row in df.iterrows():
        product_id = str(row["product_id"])
        image_url = str(row["image_url"]).strip()

        if not image_url or image_url == "nan":
            failed += 1
            continue

        ext = ".jpg"
        if ".png" in image_url.lower():
            ext = ".png"
        elif ".webp" in image_url.lower():
            ext = ".webp"

        file_path = OUTPUT_DIR / f"{safe_filename(product_id)}{ext}"

        if file_path.exists():
            downloaded += 1
            continue

        try:
            response = session.get(image_url, timeout=20)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                f.write(response.content)

            downloaded += 1

        except Exception as e:
            failed += 1
            print(f"Failed for {product_id}: {e}")

    print(f"Downloaded: {downloaded}")
    print(f"Failed: {failed}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    download_images()