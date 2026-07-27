from __future__ import annotations

import pandas as pd
from app.agents.base_agent import BaseAgent
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent
from app.config.paths import SENTIMENT_FEATURES_PATH

logger = get_logger("agents.sentiment")

_REQUIRED_COLS = [
    "product_id", "avg_sentiment_score",
    "positive_review_ratio", "neutral_review_ratio",
    "negative_review_ratio",
]

_SENTIMENT_DF: pd.DataFrame | None = None  # shared across every SentimentAgent instance


def _load_sentiment_df() -> pd.DataFrame:
    """Loads and indexes the sentiment table once per process, shared
    across every consumer -- not just every SentimentAgent instance.

    CompetitiveService constructs its own SentimentAgent rather than
    sharing the orchestrator's, same as it does for DataAgent -- without
    this, the same CSV gets read and parsed twice. See _load_features_df
    in data_agent.py for the same pattern and the same not-thread-safe-
    against-the-very-first-race caveat.
    """
    global _SENTIMENT_DF
    if _SENTIMENT_DF is None:
        try:
            df = pd.read_csv(SENTIMENT_FEATURES_PATH)
        except Exception:
            logger.error("SentimentAgent: failed to load sentiment CSV", exc_info=True)
            raise

        for col in _REQUIRED_COLS:
            if col not in df.columns:
                raise RuntimeError(f"SentimentAgent: missing column '{col}' in sentiment CSV")

        df["product_id"] = df["product_id"].astype(str)
        _SENTIMENT_DF = df.set_index("product_id", drop=False)

    return _SENTIMENT_DF


def _safe_float(value, default: float = 0.0) -> float:
    # pd.isna() catches NaN and None correctly -- "value or default" does
    # not: NaN is truthy in Python (bool(float("nan")) is True), so
    # "value or default" returns the NaN right back instead of the
    # default. Confirmed: a product with a genuinely missing sentiment
    # score (e.g. zero reviews yet) was returning NaN for every field
    # instead of the 0.0 this was clearly meant to guarantee.
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


_EMPTY_RESULT = {
    "avg_sentiment_score": 0.0,
    "positive_review_ratio": 0.0,
    "neutral_review_ratio": 0.0,
    "negative_review_ratio": 0.0,
}


class SentimentAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="SentimentAgent")
        # Shared, plain-indexed DataFrame -- do not mutate in place.
        self.sentiment_df = _load_sentiment_df()

    @traced_agent("SentimentAgent.run")
    def run(self, product_id: str) -> dict:
        product_id = str(product_id)

        try:
            row = self.sentiment_df.loc[product_id]
        except KeyError:
            return dict(_EMPTY_RESULT)

        if isinstance(row, pd.DataFrame):
            logger.warning(
                f"{self.name}: multiple rows found for product_id={product_id}, "
                f"using the first"
            )
            row = row.iloc[0]

        return {
            "avg_sentiment_score": _safe_float(row.get("avg_sentiment_score")),
            "positive_review_ratio": _safe_float(row.get("positive_review_ratio")),
            "neutral_review_ratio": _safe_float(row.get("neutral_review_ratio")),
            "negative_review_ratio": _safe_float(row.get("negative_review_ratio")),
        }