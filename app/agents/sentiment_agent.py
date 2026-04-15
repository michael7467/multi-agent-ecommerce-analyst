from __future__ import annotations

import pandas as pd

from app.agents.base_agent import BaseAgent


SENTIMENT_FEATURES_PATH = "data/processed/electronics_sentiment_features.csv"


class SentimentAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="SentimentAgent")
        self.sentiment_df = pd.read_csv(SENTIMENT_FEATURES_PATH)

    def run(self, product_id: str) -> dict:
        matches = self.sentiment_df[
            self.sentiment_df["product_id"].astype(str) == str(product_id)
        ]

        if matches.empty:
            return {
                "avg_sentiment_score": 0.0,
                "positive_review_ratio": 0.0,
                "neutral_review_ratio": 0.0,
                "negative_review_ratio": 0.0,
            }

        row = matches.iloc[0]

        return {
            "avg_sentiment_score": float(row["avg_sentiment_score"]),
            "positive_review_ratio": float(row["positive_review_ratio"]),
            "neutral_review_ratio": float(row["neutral_review_ratio"]),
            "negative_review_ratio": float(row["negative_review_ratio"]),
        }