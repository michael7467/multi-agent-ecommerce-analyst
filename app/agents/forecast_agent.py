from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.models.forecasting.predict import PricePredictor
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent

logger = get_logger("agents.forecast")

class ForecastAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="ForecastAgent")
        self.predictor = PricePredictor()

    @traced_agent
    def run(self, product_data: dict) -> dict:
        if not isinstance(product_data, dict):
            raise ValueError("ForecastAgent: product_data must be a dict")

        required = [
            "review_count", "avg_rating", "rating_std",
            "verified_purchase_ratio", "avg_review_length",
            "review_time_span"
        ]

        for key in required:
            if key not in product_data:
                raise ValueError(f"ForecastAgent: missing required field '{key}'")

        model_input = {
            "review_count": float(product_data["review_count"]),
            "avg_rating": float(product_data["avg_rating"]),
            "rating_std": float(product_data["rating_std"]),
            "verified_purchase_ratio": float(product_data["verified_purchase_ratio"]),
            "avg_review_length": float(product_data["avg_review_length"]),
            "review_time_span": float(product_data["review_time_span"]),
            "title": product_data.get("title", ""),
            "categories": product_data.get("categories", ""),
        }

        try:
            prediction = self.predictor.predict(model_input)
        except Exception:
            logger.error(f"{self.name}: forecast failed", exc_info=True)
            raise

        return {"predicted_class": prediction["predicted_class"]}
