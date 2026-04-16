from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.agents.base_agent import BaseAgent


TOPIC_KEYWORDS_PATH = Path("artifacts/topic_modeling/topic_keywords_global.csv")


class TopicAgent(BaseAgent):
    def __init__(self, topic_keywords_path: str | Path = TOPIC_KEYWORDS_PATH) -> None:
        super().__init__(name="TopicAgent")
        self.topic_keywords_path = Path(topic_keywords_path)

        if not self.topic_keywords_path.exists():
            raise FileNotFoundError(
                f"Topic keywords file not found: {self.topic_keywords_path}"
            )

        self.topic_df = pd.read_csv(self.topic_keywords_path)

        # remove BERTopic outlier topic
        if "topic_id" in self.topic_df.columns:
            self.topic_df = self.topic_df[self.topic_df["topic_id"] != -1].copy()

        self.topic_df = self.topic_df.reset_index(drop=True)

    def _extract_pain_points(self, df: pd.DataFrame, top_k: int = 5) -> list[dict]:
        pain_keywords = [
            "issue", "issues", "problem", "problems", "broken", "broke",
            "bad", "poor", "damage", "damaged", "defect", "defective",
            "return", "refund", "weak", "noise", "hollow", "disconnect",
            "slow", "fail", "failed"
        ]

        pain_points = []
        for _, row in df.iterrows():
            keywords = str(row.get("keywords", "")).lower()
            if any(term in keywords for term in pain_keywords):
                pain_points.append(
                    {
                        "topic_id": int(row["topic_id"]),
                        "topic_name": row.get("topic_name", ""),
                        "count": int(row.get("count", 0)),
                        "keywords": row.get("keywords", ""),
                    }
                )

            if len(pain_points) >= top_k:
                break

        return pain_points

    def run(self, top_k: int = 5) -> dict:
        df = self.topic_df.copy()

        if "count" in df.columns:
            df = df.sort_values(by="count", ascending=False)

        top_themes = []
        for _, row in df.head(top_k).iterrows():
            top_themes.append(
                {
                    "topic_id": int(row["topic_id"]),
                    "topic_name": row.get("topic_name", ""),
                    "count": int(row.get("count", 0)),
                    "keywords": row.get("keywords", ""),
                }
            )

        pain_points = self._extract_pain_points(df, top_k=top_k)

        return {
            "top_themes": top_themes,
            "pain_points": pain_points,
        }


if __name__ == "__main__":
    agent = TopicAgent()
    result = agent.run(top_k=5)

    print("\n=== TOP THEMES ===\n")
    for item in result["top_themes"]:
        print(item)

    print("\n=== PAIN POINTS ===\n")
    for item in result["pain_points"]:
        print(item)