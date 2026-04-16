from __future__ import annotations

from copy import deepcopy

from app.models.forecasting.predict import PricePredictor


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

    def _validate_product_data(self, product_data: dict) -> None:
        missing = [feature for feature in self.REQUIRED_FEATURES if feature not in product_data]
        if missing:
            raise ValueError(
                f"Counterfactual analysis requires these missing fields: {missing}"
            )

    def _build_model_input(self, product_data: dict) -> dict:
        return {
            "review_count": product_data["review_count"],
            "avg_rating": product_data["avg_rating"],
            "rating_std": product_data["rating_std"],
            "verified_purchase_ratio": product_data["verified_purchase_ratio"],
            "avg_review_length": product_data["avg_review_length"],
            "review_time_span": product_data["review_time_span"],
            "title": product_data.get("title", ""),
            "categories": product_data.get("categories", ""),
        }

    def _predict_class(self, product_data: dict) -> str:
        self._validate_product_data(product_data)
        model_input = self._build_model_input(product_data)
        result = self.predictor.predict(model_input)
        return str(result["predicted_class"]).lower()

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

    def _class_rank(self, class_name: str) -> int:
        return self.CLASS_ORDER.index(class_name)

    def _direction_label(self, original_class: str, new_class: str) -> str:
        if self._class_rank(new_class) > self._class_rank(original_class):
            return "upgrade"
        if self._class_rank(new_class) < self._class_rank(original_class):
            return "downgrade"
        return "same"

    def _generate_candidate_changes(self, original_class: str) -> list[tuple[str, list[float]]]:
        upward_features = [
            ("avg_rating", [0.2, 0.4, 0.6, 0.8]),
            ("review_count", [5, 10, 20, 50]),
            ("verified_purchase_ratio", [0.05, 0.10, 0.20]),
            ("avg_review_length", [20, 50, 100]),
        ]

        downward_features = [
            ("avg_rating", [-0.2, -0.4, -0.6, -0.8]),
            ("review_count", [-5, -10, -20, -50]),
            ("verified_purchase_ratio", [-0.05, -0.10, -0.20]),
            ("avg_review_length", [-20, -50, -100]),
        ]

        if original_class == "low":
            return upward_features
        if original_class == "high":
            return downward_features

        # for mid, try both directions
        combined = []
        for (feature_up, up_deltas), (_, down_deltas) in zip(upward_features, downward_features):
            combined.append((feature_up, up_deltas + down_deltas))
        return combined

    def _search_counterfactuals(self, product_data: dict) -> list[dict]:
        original_class = self._predict_class(product_data)
        candidate_changes = self._generate_candidate_changes(original_class)

        counterfactuals: list[dict] = []

        for feature, deltas in candidate_changes:
            for delta in deltas:
                modified = deepcopy(product_data)

                original_value = float(modified.get(feature, 0.0))
                new_value = self._clip_feature(feature, original_value + delta)
                modified[feature] = new_value

                new_class = self._predict_class(modified)

                if new_class != original_class:
                    change_type = self._direction_label(original_class, new_class)

                    counterfactuals.append(
                        {
                            "feature": feature,
                            "original_value": product_data.get(feature),
                            "new_value": new_value,
                            "original_class": original_class,
                            "new_class": new_class,
                            "change_type": change_type,
                            "delta": delta,
                            "explanation": (
                                f"If {feature} changed from "
                                f"{product_data.get(feature)} to {new_value}, "
                                f"the model would predict '{new_class}' instead of '{original_class}'."
                            ),
                        }
                    )
                    break

        return counterfactuals

    def generate_counterfactuals(self, product_data: dict) -> list[dict]:
        self._validate_product_data(product_data)

        original_class = self._predict_class(product_data)
        counterfactuals = self._search_counterfactuals(product_data)

        if counterfactuals:
            return counterfactuals

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