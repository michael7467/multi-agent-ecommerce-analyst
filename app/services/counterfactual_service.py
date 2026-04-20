from __future__ import annotations

from copy import deepcopy
from app.models.forecasting.predict import PricePredictor
from app.logging.logger import get_logger
from app.observability.tracing import get_tracer

logger = get_logger("counterfactual.service")


class CounterfactualService:
    REQUIRED_FEATURES = [
        "review_count",
        "avg_rating",
        "rating_std",
        "verified_purchase_ratio",
        "avg_review_length",
        "review_time_span",
    ]

    CLASS_ORDER = ["low", "mid", "high"]

    def __init__(self) -> None:
        self.predictor = PricePredictor()
        self.tracer = get_tracer("app.counterfactual_service")

    # -----------------------------
    # Validation
    # -----------------------------
    def _validate_product_data(self, product_data: dict) -> None:
        missing = [f for f in self.REQUIRED_FEATURES if f not in product_data]
        if missing:
            logger.error(f"Missing required fields: {missing}")
            raise ValueError(f"Counterfactual analysis requires these missing fields: {missing}")

    # -----------------------------
    # Prediction
    # -----------------------------
    def _predict_class(self, product_data: dict) -> str:
        with self.tracer.start_as_current_span("counterfactual.predict_class") as span:
            self._validate_product_data(product_data)

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

            result = self.predictor.predict(model_input)
            predicted = str(result["predicted_class"]).lower()

            span.set_attribute("predicted_class", predicted)
            logger.debug(f"Predicted class: {predicted}")

            return predicted

    # -----------------------------
    # Feature clipping
    # -----------------------------
    def _clip_feature(self, feature: str, value: float) -> float:
        if feature == "avg_rating":
            return max(1.0, min(5.0, value))
        if feature == "verified_purchase_ratio":
            return max(0.0, min(1.0, value))
        if feature in ["review_count", "avg_review_length", "review_time_span"]:
            return max(0.0, value)
        if feature == "rating_std":
            return max(0.0, value)
        return value

    # -----------------------------
    # Counterfactual search
    # -----------------------------
    def _search_counterfactuals(self, product_data: dict) -> list[dict]:
        with self.tracer.start_as_current_span("counterfactual.search") as span:
            original_class = self._predict_class(product_data)
            span.set_attribute("original_class", original_class)

            candidate_changes = self._generate_candidate_changes(original_class)
            counterfactuals: list[dict] = []

            for feature, deltas in candidate_changes:
                for delta in deltas:
                    logger.debug(f"Testing feature={feature}, delta={delta}")

                    modified = deepcopy(product_data)
                    original_value = float(modified.get(feature, 0.0))
                    new_value = self._clip_feature(feature, original_value + delta)
                    modified[feature] = new_value

                    new_class = self._predict_class(modified)

                    if new_class != original_class:
                        logger.info(
                            f"Counterfactual found: {feature} {original_value}→{new_value} "
                            f"({original_class} → {new_class})"
                        )
                        span.add_event(
                            "counterfactual_found",
                            {
                                "feature": feature,
                                "delta": delta,
                                "new_class": new_class,
                            },
                        )

                        counterfactuals.append(
                            {
                                "feature": feature,
                                "original_value": product_data.get(feature),
                                "new_value": new_value,
                                "original_class": original_class,
                                "new_class": new_class,
                                "change_type": self._direction_label(original_class, new_class),
                                "delta": delta,
                                "explanation": (
                                    f"If {feature} changed from {product_data.get(feature)} "
                                    f"to {new_value}, the model would predict '{new_class}' "
                                    f"instead of '{original_class}'."
                                ),
                            }
                        )
                        break

            return counterfactuals

    # -----------------------------
    # Public API
    # -----------------------------
    def generate_counterfactuals(self, product_data: dict) -> list[dict]:
        with self.tracer.start_as_current_span("counterfactual.generate") as span:
            logger.info("Starting counterfactual generation")
            self._validate_product_data(product_data)

            original_class = self._predict_class(product_data)
            span.set_attribute("original_class", original_class)

            counterfactuals = self._search_counterfactuals(product_data)

            if counterfactuals:
                logger.info(f"Generated {len(counterfactuals)} counterfactuals")
                return counterfactuals

            logger.info("No counterfactuals found; returning fallback explanation")

            if original_class == "high":
                return [
                    {
                        "feature": None,
                        "original_value": None,
                        "new_value": None,
                        "original_class": original_class,
                        "new_class": original_class,
                        "change_type": "none",
                        "delta": 0.0,
                        "explanation": (
                            "This product is already classified in the highest price class, "
                            "and no simple feature change tested here produced a different class."
                        ),
                    }
                ]

            if original_class == "low":
                return [
                    {
                        "feature": None,
                        "original_value": None,
                        "new_value": None,
                        "original_class": original_class,
                        "new_class": original_class,
                        "change_type": "none",
                        "delta": 0.0,
                        "explanation": (
                            "No simple feature change tested here was enough to move this product "
                            "into a higher price class."
                        ),
                    }
                ]

            return [
                {
                    "feature": None,
                    "original_value": None,
                    "new_value": None,
                    "original_class": original_class,
                    "new_class": original_class,
                    "change_type": "none",
                    "delta": 0.0,
                    "explanation": (
                        "No simple upward or downward feature change tested here produced a different class."
                    ),
                }
            ]