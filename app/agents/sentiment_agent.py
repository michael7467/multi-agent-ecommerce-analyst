from __future__ import annotations

import pandas as pd
from app.agents.base_agent import BaseAgent
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent
from app.config.paths import SENTIMENT_FEATURES_PATH

logger = get_logger("agents.sentiment")

class SentimentAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="SentimentAgent")

        try:
            self.sentiment_df = pd.read_csv(SENTIMENT_FEATURES_PATH)
        except Exception:
            logger.error(f"{self.name}: failed to load sentiment CSV", exc_info=True)
            raise

        required_cols = [
            "product_id", "avg_sentiment_score",
            "positive_review_ratio", "neutral_review_ratio",
            "negative_review_ratio"
        ]

        for col in required_cols:
            if col not in self.sentiment_df.columns:
                raise RuntimeError(f"SentimentAgent: missing column '{col}' in sentiment CSV")

        self.cache = {}

    @traced_agent
    def run(self, product_id: str) -> dict:
        if not isinstance(product_id, str):
            raise ValueError("SentimentAgent: product_id must be a string")

        if product_id in self.cache:
            return self.cache[product_id]

        matches = self.sentiment_df[
            self.sentiment_df["product_id"].astype(str) == str(product_id)
        ]

        if matches.empty:
            result = {
                "avg_sentiment_score": 0.0,
                "positive_review_ratio": 0.0,
                "neutral_review_ratio": 0.0,
                "negative_review_ratio": 0.0,
            }
            self.cache[product_id] = result
            return result

        row = matches.iloc[0]

        result = {
            "avg_sentiment_score": float(row.get("avg_sentiment_score", 0) or 0),
            "positive_review_ratio": float(row.get("positive_review_ratio", 0) or 0),
            "neutral_review_ratio": float(row.get("neutral_review_ratio", 0) or 0),
            "negative_review_ratio": float(row.get("negative_review_ratio", 0) or 0),
        }

        self.cache[product_id] = result
        return result
