from __future__ import annotations

import math

from app.agents.base_agent import BaseAgent
from app.models.forecasting.predict import PricePredictor
from app.observability.agent_tracing import traced_agent

class ForecastAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="ForecastAgent")
        self.predictor = PricePredictor()

    @traced_agent("ForecastAgent.run")
    def run(self, product_data: dict) -> dict:
        if not isinstance(product_data, dict):
            raise ValueError("ForecastAgent: product_data must be a dict")

        required = [
            "review_count", "avg_rating", "rating_std",
            "verified_purchase_ratio", "avg_review_length",
            "review_time_span"
        ]

        model_input = {}
        for key in required:
            if key not in product_data:
                raise ValueError(f"ForecastAgent: missing required field '{key}'")

            raw_value = product_data[key]
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"ForecastAgent: field '{key}' is not numeric (got {raw_value!r})"
                )

            if math.isnan(value):
                # Field presence alone doesn't catch this: a NaN feature
                # value (e.g. from DataAgent's source CSV having a genuine
                # gap for this product) still passes "key in product_data",
                # since the key IS there, just with a NaN value. float(nan)
                # doesn't raise either, so this would otherwise reach the
                # model as a silent NaN input with unknown behavior --
                # nothing in PricePredictor.predict() imputes or rejects it
                # explicitly.
                raise ValueError(f"ForecastAgent: field '{key}' is NaN")

            model_input[key] = value

        model_input["title"] = product_data.get("title", "")
        model_input["categories"] = product_data.get("categories", "")

        prediction = self.predictor.predict(model_input)
        return {"predicted_class": prediction["predicted_class"]}