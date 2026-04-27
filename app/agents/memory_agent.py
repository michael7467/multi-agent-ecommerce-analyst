from __future__ import annotations

from typing import Optional
from app.agents.base_agent import BaseAgent
from app.memory.db import get_connection
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent

logger = get_logger("agents.memory")

class MemoryAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="MemoryAgent")

    @traced_agent("MemoryAgent.run")
    def run(self, product_id: str) -> dict:
        if not isinstance(product_id, str):
            raise ValueError("MemoryAgent: product_id must be a string")

        try:
            memory = self.get_product_memory(product_id)
        except Exception:
            logger.error(f"{self.name}: failed to fetch memory", exc_info=True)
            raise

        return {"memory": memory}

    def get_product_memory(self, product_id: str) -> Optional[dict]:
        with get_connection() as conn:
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

        if not row:
            return None

        return {
            "product_id": row[0],
            "title": row[1],
            "last_predicted_class": row[2],
            "avg_sentiment": row[3],
            "last_report": row[4],
        }

    @traced_agent("MemoryAgent.save_product_memory")
    def save_product_memory(self, analysis_result: dict) -> None:
        product_id = analysis_result.get("product_id")
        title = analysis_result.get("title", "")
        report = analysis_result.get("report")

        predicted_class = (
            analysis_result.get("predicted_class")
            or analysis_result.get("last_predicted_class")
            or analysis_result.get("price_class")
            or analysis_result.get("forecast", {}).get("predicted_class")
        )

        if not product_id or not report:
            logger.warning(
                "MemoryAgent: skipping memory save because product_id or report is missing"
            )
            return

        if predicted_class is None:
            logger.warning(
                "MemoryAgent: skipping memory save because predicted_class is missing",
                extra={"product_id": product_id},
            )
            return

        sentiment = analysis_result.get("sentiment", {})
        avg_sentiment = sentiment.get("avg_sentiment_score")

        with get_connection() as conn:
            cursor = conn.cursor()
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
                    product_id,
                    title,
                    predicted_class,
                    avg_sentiment,
                    report,
                ),
            )
            conn.commit()

    @traced_agent("MemoryAgent.save_history")
    def save_history(self, product_id: str, query: str, report: str) -> None:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO analysis_history (product_id, query, report)
                VALUES (?, ?, ?)
                """,
                (product_id, query, report),
            )
