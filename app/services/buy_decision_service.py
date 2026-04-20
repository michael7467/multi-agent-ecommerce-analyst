from __future__ import annotations

from app.logging.logger import get_logger
from app.observability.tracing import get_tracer

logger = get_logger("buy_decision.service")


class BuyDecisionService:
    def __init__(self) -> None:
        self.tracer = get_tracer("app.buy_decision")

    def make_decision(self, analysis_result: dict) -> dict:
        if not isinstance(analysis_result, dict):
            raise ValueError("analysis_result must be a dictionary")

        with self.tracer.start_as_current_span("buy_decision.make_decision") as span:
            title = analysis_result.get("title", "")
            price = analysis_result.get("price")
            predicted_class = str(analysis_result.get("predicted_class", "")).lower()
            sentiment = analysis_result.get("sentiment", {})
            aspect_sentiment = analysis_result.get("aspect_sentiment", {})
            evidence = analysis_result.get("evidence", [])
            recommendations = analysis_result.get("recommendations", [])

            avg_sentiment = float(sentiment.get("avg_sentiment_score", 0.0))
            positive_ratio = float(sentiment.get("positive_review_ratio", 0.0))

            def norm(label: str) -> str:
                label = str(label).lower()
                return label if label in {"positive", "negative", "mixed"} else "mixed"

            sound = norm(aspect_sentiment.get("sound_quality", {}).get("label", "mixed"))
            battery = norm(aspect_sentiment.get("battery_life", {}).get("label", "mixed"))
            comfort = norm(aspect_sentiment.get("comfort", {}).get("label", "mixed"))
            build_quality = norm(aspect_sentiment.get("build_quality", {}).get("label", "mixed"))
            price_value = norm(aspect_sentiment.get("price_value", {}).get("label", "mixed"))

            pros, cons = [], []
            recommended_for, not_recommended_for = [], []

            # --- Global sentiment ---
            if avg_sentiment >= 0.75 or positive_ratio >= 0.75:
                pros.append("overall customer sentiment is strongly positive")
            elif avg_sentiment <= 0.40:
                cons.append("overall customer sentiment is weak or mixed")

            # --- Comfort ---
            if comfort == "positive":
                pros.append("comfort is a clear strength")
                recommended_for.append("users who prioritize comfort for home or office use")
            elif comfort == "negative":
                cons.append("comfort may be a weakness")
                not_recommended_for.append("users who need long-session comfort")

            # --- Battery ---
            if battery == "positive":
                pros.append("battery life is positively perceived")
                recommended_for.append("users who want longer wireless use")
            elif battery == "negative":
                cons.append("battery life may disappoint some users")
                not_recommended_for.append("users who strongly depend on long battery life")

            # --- Sound ---
            if sound == "positive":
                pros.append("sound quality is generally appreciated")
                recommended_for.append("users who want solid everyday audio quality")
            else:
                cons.append("sound quality is mixed and may not satisfy everyone")
                not_recommended_for.append("users who care deeply about premium sound fidelity")

            # --- Build quality ---
            if build_quality == "negative":
                cons.append("build quality may not feel strong enough for some users")

            # --- Price-value ---
            if price_value == "positive":
                pros.append("the product appears to offer good price-value balance")
                recommended_for.append("value-conscious users")
            elif price_value == "negative":
                cons.append("the price-value balance may be questionable")

            # --- Price tier ---
            if predicted_class == "high":
                cons.append("the product is positioned in a higher price tier")
                not_recommended_for.append("users on a tight budget")
            elif predicted_class == "low":
                pros.append("the product is positioned in a lower price tier")
                recommended_for.append("budget-conscious users")

            # --- Alternatives ---
            if recommendations:
                cons.append("there are alternative products available in the same category")

            # --- Fallbacks ---
            if not pros:
                pros.append("the product has some potentially useful features")
            if not cons:
                cons.append("there are no major severe drawbacks clearly supported by current evidence")

            decision = self._decide_label(
                avg_sentiment=avg_sentiment,
                sound=sound,
                comfort=comfort,
                battery=battery,
                price_value=price_value,
            )

            summary = self._build_summary(
                title=title,
                price=price,
                decision=decision,
                recommended_for=recommended_for,
                not_recommended_for=not_recommended_for,
            )

            return {
                "decision": decision,
                "summary": summary,
                "pros": self._unique(pros),
                "cons": self._unique(cons),
                "recommended_for": self._unique(recommended_for),
                "not_recommended_for": self._unique(not_recommended_for),
                "evidence_count": len(evidence),
            }

    def _decide_label(self, avg_sentiment, sound, comfort, battery, price_value) -> str:
        strong_positive = sum(label == "positive" for label in [sound, comfort, battery, price_value])
        weak_or_negative = sum(label in {"negative", "mixed"} for label in [sound, comfort, battery, price_value])

        if avg_sentiment >= 0.75 and strong_positive >= 2:
            return "recommended"
        if avg_sentiment < 0.45 or weak_or_negative >= 3:
            return "not recommended"
        return "conditionally recommended"

    def _build_summary(self, title, price, decision, recommended_for, not_recommended_for) -> str:
        try:
            price_text = f" at ${float(price):.2f}"
        except Exception:
            price_text = ""

        rec_for = recommended_for[0] if recommended_for else "some general users"
        not_for = not_recommended_for[0] if not_recommended_for else "users with stricter premium expectations"

        if decision == "recommended":
            return f"{title}{price_text} is recommended, especially for {rec_for}, but may be less suitable for {not_for}."

        if decision == "not recommended":
            return f"{title}{price_text} is not strongly recommended based on the current signals, especially for users who need stronger performance in key areas."

        return f"{title}{price_text} is conditionally recommended: it may suit {rec_for}, but may not be the best choice for {not_for}."

    def _unique(self, items: list[str]) -> list[str]:
        seen, result = set(), []
        for item in items:
            if item and item not in seen:
                seen.add(item)
                result.append(item)
        return result
