from __future__ import annotations

from typing import Optional

from app.agents.base_agent import BaseAgent
from app.memory.db import get_connection


class MemoryAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="MemoryAgent")

    def run(self, product_id: str) -> dict:
        memory = self.get_product_memory(product_id)
        return {"memory": memory}

    def get_product_memory(self, product_id: str) -> Optional[dict]:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT product_id, title, last_predicted_class, avg_sentiment, last_report
            FROM product_memory
            WHERE product_id = ?
            """,
            (product_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return {
            "product_id": row[0],
            "title": row[1],
            "last_predicted_class": row[2],
            "avg_sentiment": row[3],
            "last_report": row[4],
        }

    def save_product_memory(self, analysis_result: dict) -> None:
        conn = get_connection()
        cursor = conn.cursor()

        sentiment = analysis_result.get("sentiment", {})
        avg_sentiment = sentiment.get("avg_sentiment_score")

        cursor.execute(
            """
            INSERT INTO product_memory (
                product_id, title, last_predicted_class, avg_sentiment, last_report
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(product_id) DO UPDATE SET
                title = excluded.title,
                last_predicted_class = excluded.last_predicted_class,
                avg_sentiment = excluded.avg_sentiment,
                last_report = excluded.last_report,
                last_updated = CURRENT_TIMESTAMP
            """,
            (
                analysis_result.get("product_id"),
                analysis_result.get("title"),
                analysis_result.get("predicted_class"),
                avg_sentiment,
                analysis_result.get("report"),
            ),
        )

        conn.commit()
        conn.close()

    def save_history(self, product_id: str, query: str, report: str) -> None:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO analysis_history (product_id, query, report)
            VALUES (?, ?, ?)
            """,
            (product_id, query, report),
        )

        conn.commit()
        conn.close()