from __future__ import annotations

from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired


REVIEWS_PATH = Path("data/interim/reviews_electronics_clean.csv")
OUTPUT_DIR = Path("artifacts/topic_modeling")


class TopicModelingService:
    def __init__(self) -> None:
        representation_model = KeyBERTInspired()

        self.topic_model = BERTopic(
            representation_model=representation_model,
            calculate_probabilities=False,
            verbose=True,
        )

    def load_reviews(
        self,
        input_path: str | Path = REVIEWS_PATH,
        product_id: str | None = None,
        min_review_length: int = 20,
        max_docs: int | None = 5000,
    ) -> pd.DataFrame:
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Missing reviews file: {input_path}")

        df = pd.read_csv(input_path)

        if "review_text" not in df.columns:
            raise ValueError("Expected 'review_text' column in reviews file.")

        df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()
        df = df[df["review_text"].str.len() >= min_review_length].copy()

        if product_id is not None:
            df = df[df["product_id"].astype(str) == str(product_id)].copy()

        if max_docs is not None and len(df) > max_docs:
            df = df.head(max_docs).copy()

        df = df.reset_index(drop=True)
        return df

    def fit_topics(
        self,
        reviews_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        docs = reviews_df["review_text"].tolist()

        topics, _ = self.topic_model.fit_transform(docs)

        doc_info = self.topic_model.get_document_info(docs)
        topic_info = self.topic_model.get_topic_info()

        reviews_with_topics = reviews_df.copy()
        reviews_with_topics["topic"] = topics

        if "Topic" in doc_info.columns:
            reviews_with_topics["topic"] = doc_info["Topic"].values

        return reviews_with_topics, topic_info

    def extract_topic_keywords(self, topic_info: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for _, row in topic_info.iterrows():
            topic_id = int(row["Topic"])
            topic_name = row.get("Name", "")
            count = int(row.get("Count", 0))

            if topic_id == -1:
                keywords = []
            else:
                topic_terms = self.topic_model.get_topic(topic_id) or []
                keywords = [term for term, _ in topic_terms[:10]]

            rows.append(
                {
                    "topic_id": topic_id,
                    "topic_name": topic_name,
                    "count": count,
                    "keywords": ", ".join(keywords),
                }
            )

        return pd.DataFrame(rows)

    def run(
        self,
        product_id: str | None = None,
        max_docs: int | None = 5000,
    ) -> dict[str, pd.DataFrame]:
        reviews_df = self.load_reviews(product_id=product_id, max_docs=max_docs)
        reviews_with_topics, topic_info = self.fit_topics(reviews_df)
        topic_keywords = self.extract_topic_keywords(topic_info)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        suffix = f"_{product_id}" if product_id else "_global"

        reviews_path = OUTPUT_DIR / f"reviews_with_topics{suffix}.csv"
        topic_info_path = OUTPUT_DIR / f"topic_info{suffix}.csv"
        topic_keywords_path = OUTPUT_DIR / f"topic_keywords{suffix}.csv"

        reviews_with_topics.to_csv(reviews_path, index=False)
        topic_info.to_csv(topic_info_path, index=False)
        topic_keywords.to_csv(topic_keywords_path, index=False)

        print(f"Saved reviews with topics to: {reviews_path}")
        print(f"Saved topic info to: {topic_info_path}")
        print(f"Saved topic keywords to: {topic_keywords_path}")

        return {
            "reviews_with_topics": reviews_with_topics,
            "topic_info": topic_info,
            "topic_keywords": topic_keywords,
        }


if __name__ == "__main__":
    service = TopicModelingService()

    result = service.run(
        product_id=None,   # set a product_id to do per-product topics
        max_docs=3000,     # start smaller for faster testing
    )

    print("\n=== TOPIC INFO ===")
    print(result["topic_info"].head())

    print("\n=== TOPIC KEYWORDS ===")
    print(result["topic_keywords"].head())