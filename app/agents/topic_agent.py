from __future__ import annotations

from pathlib import Path
import pandas as pd

from app.agents.base_agent import BaseAgent
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent
from app.config.paths import TOPIC_KEYWORDS_PATH


logger = get_logger("agents.topic")

class TopicAgent(BaseAgent):
    def __init__(self, topic_keywords_path: str | Path = TOPIC_KEYWORDS_PATH) -> None:
        super().__init__(name="TopicAgent")
        self.topic_keywords_path = Path(topic_keywords_path)

        if not self.topic_keywords_path.exists():
            raise FileNotFoundError(
                f"Topic keywords file not found: {self.topic_keywords_path}"
            )

        try:
            self.topic_df = pd.read_csv(self.topic_keywords_path)
        except Exception:
            logger.error(f"{self.name}: failed to load topic keywords", exc_info=True)
            raise

        required_cols = ["topic_id", "topic_name", "count", "keywords"]
        for col in required_cols:
            if col not in self.topic_df.columns:
                raise RuntimeError(f"TopicAgent: missing column '{col}' in topic CSV")

        # remove BERTopic outlier topic
        self.topic_df = self.topic_df[self.topic_df["topic_id"] != -1].copy()
        self.topic_df = self.topic_df.reset_index(drop=True)

    def _extract_pain_points(self, df: pd.DataFrame, top_k: int = 5) -> list[dict]:
        pain_keywords = [
            "issue", "issues", "problem", "problems", "broken", "broke",
            "bad", "poor", "damage", "damaged", "defect", "defective",
            "return", "refund", "weak", "noise", "hollow", "disconnect",
            "slow", "fail", "failed"
        ]

        df = df.copy()
        df["is_pain"] = df["keywords"].str.lower().apply(
            lambda kw: any(term in kw for term in pain_keywords)
        )

        pain_df = df[df["is_pain"]].sort_values(by="count", ascending=False)

        return [
            {
                "topic_id": int(row["topic_id"]),
                "topic_name": row.get("topic_name", ""),
                "count": int(row.get("count", 0)),
                "keywords": row.get("keywords", ""),
            }
            for _, row in pain_df.head(top_k).iterrows()
        ]

    @traced_agent
    def run(self, top_k: int = 5) -> dict:
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("TopicAgent: top_k must be a positive integer")

        df = self.topic_df.copy()

        if "count" in df.columns:
            df = df.sort_values(by="count", ascending=False)

        top_themes = [
            {
                "topic_id": int(row["topic_id"]),
                "topic_name": row.get("topic_name", ""),
                "count": int(row.get("count", 0)),
                "keywords": row.get("keywords", ""),
            }
            for _, row in df.head(top_k).iterrows()
        ]

        pain_points = self._extract_pain_points(df, top_k=top_k)

        return {
            "top_themes": top_themes,
            "pain_points": pain_points,
        }
