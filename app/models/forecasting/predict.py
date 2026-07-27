from __future__ import annotations

import os
import joblib
import pandas as pd


# Every other data path in this codebase goes through app.config.paths and
# is env-overridable (FEATURES_PATH, SENTIMENT_FEATURES_PATH, etc.). These
# two were the exception -- hardcoded relative paths, which only resolve
# correctly if the process's current working directory happens to match,
# and can't be pointed elsewhere without a code change. Making them
# env-overridable here at minimum; the real fix is moving these into
# app.config.paths alongside everything else for consistency.
MODEL_PATH = os.environ.get(
    "PRICE_MODEL_PATH", "artifacts/models/price_class_model_with_text.joblib"
)
ENCODER_PATH = os.environ.get(
    "PRICE_ENCODER_PATH", "artifacts/models/price_class_label_encoder_with_text.joblib"
)


class PricePredictor:
    def __init__(self):
        # joblib.load() raises a bare FileNotFoundError/UnpicklingError with
        # no context otherwise -- and since this runs at construction time,
        # inside ForecastAgent.__init__(), inside the orchestrator's own
        # __init__, a missing or corrupted model file currently takes down
        # the entire app at startup, not just the forecast feature. Not
        # changing that fail-fast-at-startup behavior here -- that's a real
        # design decision (fail loud at deploy time vs. lazy-load and
        # degrade gracefully like the rest of the pipeline now does) that's
        # bigger than this file and worth deciding deliberately, not by
        # default. This just makes the failure diagnosable when it happens.
        try:
            self.model = joblib.load(MODEL_PATH)
            self.encoder = joblib.load(ENCODER_PATH)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"PricePredictor: model artifact not found ({exc}). "
                f"Expected model at '{MODEL_PATH}' and encoder at "
                f"'{ENCODER_PATH}' -- check these exist relative to the "
                f"process's working directory, or set PRICE_MODEL_PATH / "
                f"PRICE_ENCODER_PATH to override."
            ) from exc

    def predict(self, input_data: dict) -> dict:
        """
        input_data = {
            "review_count": ...,
            "avg_rating": ...,
            "rating_std": ...,
            "verified_purchase_ratio": ...,
            "avg_review_length": ...,
            "review_time_span": ...,
            "title": "...",
            "categories": "..."
        }
        """

        df = pd.DataFrame([input_data])

        pred = self.model.predict(df)[0]
        pred_label = self.encoder.inverse_transform([pred])[0]

        return {
            "predicted_class": pred_label
        }


if __name__ == "__main__":
    predictor = PricePredictor()

    sample_input = {
        "review_count": 120,
        "avg_rating": 4.5,
        "rating_std": 0.5,
        "verified_purchase_ratio": 0.9,
        "avg_review_length": 150,
        "review_time_span": 10000000,
        "title": "Wireless Noise Cancelling Headphones",
        "categories": "Electronics | Headphones"
    }

    result = predictor.predict(sample_input)

    print("Prediction:", result)