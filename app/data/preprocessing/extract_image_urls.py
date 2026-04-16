from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/raw/meta_electronics_sample.jsonl")
OUTPUT_PATH = Path("data/processed/electronics_image_urls.csv")


def pick_main_image(images: list[dict]) -> str | None:
    if not images:
        return None

    for img in images:
        if img.get("variant") == "MAIN":
            return img.get("hi_res") or img.get("large") or img.get("thumb")

    first = images[0]
    return first.get("hi_res") or first.get("large") or first.get("thumb")


def extract_image_urls() -> None:
    rows = []

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            product_id = item.get("parent_asin") or item.get("asin")
            title = item.get("title", "")
            images = item.get("images", [])

            image_url = pick_main_image(images)
            if not product_id or not image_url:
                continue

            rows.append(
                {
                    "product_id": str(product_id),
                    "title": str(title),
                    "image_url": str(image_url),
                }
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(df)} image URLs to {OUTPUT_PATH}")
    print(df.head())


if __name__ == "__main__":
    extract_image_urls()