from __future__ import annotations

from pathlib import Path

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer


class SentimentFeatureBuilder:
    def __init__(self, reviews_df: pd.DataFrame) -> None:
        self.reviews_df = reviews_df.copy()
        self.analyzer = SentimentIntensityAnalyzer()

    def _score_text(self, text: str) -> float:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        return self.analyzer.polarity_scores(text)["compound"]

    def build(self) -> pd.DataFrame:
        df = self.reviews_df.copy()

        df["review_text"] = df["review_text"].fillna("").astype(str)
        df["sentiment_score"] = df["review_text"].apply(self._score_text)

        df["sentiment_label"] = pd.cut(
            df["sentiment_score"],
            bins=[-1.0, -0.05, 0.05, 1.0],
            labels=["negative", "neutral", "positive"],
        )

        agg_df = (
            df.groupby("product_id")
            .agg(
                avg_sentiment_score=("sentiment_score", "mean"),
                positive_review_ratio=("sentiment_label", lambda x: (x == "positive").mean()),
                neutral_review_ratio=("sentiment_label", lambda x: (x == "neutral").mean()),
                negative_review_ratio=("sentiment_label", lambda x: (x == "negative").mean()),
            )
            .reset_index()
        )

        return agg_df


def save_sentiment_features(
    input_path: str = "data/interim/reviews_electronics_clean.csv",
    output_path: str = "data/processed/electronics_sentiment_features.csv",
) -> pd.DataFrame:
    reviews_df = pd.read_csv(input_path)

    builder = SentimentFeatureBuilder(reviews_df)
    sentiment_df = builder.build()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sentiment_df.to_csv(output_path, index=False)

    print(f"Saved sentiment features to: {output_path}")
    print("Shape:", sentiment_df.shape)
    print(sentiment_df.head())

    return sentiment_df


if __name__ == "__main__":
    save_sentiment_features()