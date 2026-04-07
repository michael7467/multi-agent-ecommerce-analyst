from __future__ import annotations

import gzip
import json
from pathlib import Path


RAW_DIR = Path("data/raw")
REVIEWS_INPUT = RAW_DIR / "Electronics.jsonl.gz"
META_INPUT = RAW_DIR / "meta_Electronics.jsonl.gz"

REVIEWS_OUTPUT = RAW_DIR / "reviews_electronics_sample.jsonl"
META_OUTPUT = RAW_DIR / "meta_electronics_sample.jsonl"


def create_review_sample(sample_size: int = 50000) -> set[str]:
    """
    Read a limited number of review rows from the compressed Electronics file,
    save them locally, and return the set of product IDs found.
    """
    if not REVIEWS_INPUT.exists():
        raise FileNotFoundError(f"Missing reviews file: {REVIEWS_INPUT}")

    product_ids: set[str] = set()
    written = 0

    with gzip.open(REVIEWS_INPUT, "rt", encoding="utf-8") as fin, \
         open(REVIEWS_OUTPUT, "w", encoding="utf-8") as fout:

        for line in fin:
            row = json.loads(line)

            product_id = row.get("parent_asin") or row.get("asin")
            if product_id:
                product_ids.add(str(product_id))

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

            if written >= sample_size:
                break

    print(f"Saved {written} review rows to {REVIEWS_OUTPUT}")
    print(f"Collected {len(product_ids)} unique product IDs from reviews")
    return product_ids


def create_matching_metadata_sample(product_ids: set[str]) -> None:
    """
    Scan the compressed metadata file and keep only rows whose product ID
    appears in the sampled reviews.
    """
    if not META_INPUT.exists():
        raise FileNotFoundError(f"Missing metadata file: {META_INPUT}")

    matched = 0
    scanned = 0

    with gzip.open(META_INPUT, "rt", encoding="utf-8") as fin, \
         open(META_OUTPUT, "w", encoding="utf-8") as fout:

        for line in fin:
            scanned += 1
            row = json.loads(line)

            product_id = row.get("parent_asin") or row.get("asin")
            if product_id and str(product_id) in product_ids:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                matched += 1

    print(f"Scanned {scanned} metadata rows")
    print(f"Saved {matched} matching metadata rows to {META_OUTPUT}")


if __name__ == "__main__":
    ids = create_review_sample(sample_size=50000)
    create_matching_metadata_sample(ids)