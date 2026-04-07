from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.models.forecasting.predict import PricePredictor


class ForecastAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="ForecastAgent")
        self.predictor = PricePredictor()

    def run(self, product_data: dict) -> dict:
        model_input = {
            "review_count": product_data["review_count"],
            "avg_rating": product_data["avg_rating"],
            "rating_std": product_data["rating_std"],
            "verified_purchase_ratio": product_data["verified_purchase_ratio"],
            "avg_review_length": product_data["avg_review_length"],
            "review_time_span": product_data["review_time_span"],
            "title": product_data.get("title", ""),
            "categories": product_data.get("categories", ""),
        }

        prediction = self.predictor.predict(model_input)
        return {
            "predicted_class": prediction["predicted_class"]
        }