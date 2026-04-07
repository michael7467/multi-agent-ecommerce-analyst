from datasets import load_dataset
import json
from pathlib import Path

def download_reviews(output_path: str, sample_size: str = "1%"):
    print("Downloading Electronics reviews...")

    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        name="raw_review_Electronics",
        split=f"train[:{sample_size}]",
        trust_remote_code=True,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved reviews to {output_path}")

if __name__ == "__main__":
    download_reviews("data/raw/reviews_electronics.jsonl", sample_size="1%")