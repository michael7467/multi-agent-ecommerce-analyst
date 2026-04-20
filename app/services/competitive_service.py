from __future__ import annotations

from app.agents.data_agent import DataAgent
from app.agents.sentiment_agent import SentimentAgent
from app.agents.aspect_sentiment_agent import AspectSentimentAgent
from app.agents.forecast_agent import ForecastAgent
from app.agents.recommender_agent import RecommenderAgent
from app.logging.logger import get_logger
from app.observability.tracing import get_tracer

logger = get_logger("competitive.service")


class CompetitiveService:
    def __init__(self) -> None:
        self.data_agent = DataAgent()
        self.sentiment_agent = SentimentAgent()
        self.aspect_sentiment_agent = AspectSentimentAgent(backend="zero_shot")
        self.forecast_agent = ForecastAgent()
        self.recommender_agent = RecommenderAgent()
        self.tracer = get_tracer("app.competitive_service")

    def analyze(self, product_id: str, top_k: int = 5) -> dict:
        if not isinstance(product_id, str) or not product_id.strip():
            raise ValueError("product_id must be a non-empty string")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        with self.tracer.start_as_current_span("competitive.analyze") as span:
            span.set_attribute("product_id", product_id)
            span.set_attribute("top_k", top_k)

            # --- Base product ---
            base_product = self.data_agent.run(product_id=product_id)
            base_sentiment = self.sentiment_agent.run(product_id=product_id)

            aspect_result = self.aspect_sentiment_agent.run(
                product_id=product_id,
                top_k=2,
            )
            base_aspect_sentiment = aspect_result.get("aspect_sentiment", {})

            base_forecast = self.forecast_agent.run(product_data=base_product)

            # --- Competitors ---
            recs = self.recommender_agent.run(product_id=product_id, top_k=top_k)
            recommendations = recs.get("recommendations", [])

            competitors = []

            for rec in recommendations:
                competitor_id = rec.get("product_id")
                if not competitor_id:
                    continue

                try:
                    comp_data = self.data_agent.run(product_id=competitor_id)
                    comp_sentiment = self.sentiment_agent.run(product_id=competitor_id)

                    comp_aspect = self.aspect_sentiment_agent.run(
                        product_id=competitor_id,
                        top_k=2,
                    ).get("aspect_sentiment", {})

                    comp_forecast = self.forecast_agent.run(product_data=comp_data)

                    competitors.append(
                        {
                            "product_id": competitor_id,
                            "title": comp_data.get("title", ""),
                            "price": comp_data.get("price"),
                            "predicted_class": comp_forecast.get("predicted_class"),
                            "avg_sentiment": comp_sentiment.get("avg_sentiment_score"),
                            "similarity_score": rec.get("similarity_score"),
                            "aspect_sentiment": comp_aspect,
                        }
                    )

                except Exception:
                    logger.error(f"Failed to analyze competitor {competitor_id}", exc_info=True)
                    continue

            insights = self._generate_insights(
                base_product=base_product,
                base_sentiment=base_sentiment,
                base_aspect_sentiment=base_aspect_sentiment,
                base_forecast=base_forecast,
                competitors=competitors,
            )

            return {
                "base_product": {
                    "product_id": product_id,
                    "title": base_product.get("title", ""),
                    "price": base_product.get("price"),
                    "predicted_class": base_forecast.get("predicted_class"),
                    "avg_sentiment": base_sentiment.get("avg_sentiment_score"),
                    "aspect_sentiment": base_aspect_sentiment,
                },
                "competitors": competitors,
                "insights": insights,
            }

    def _generate_insights(
        self,
        base_product: dict,
        base_sentiment: dict,
        base_aspect_sentiment: dict,
        base_forecast: dict,
        competitors: list[dict],
    ) -> list[str]:
        insights: list[str] = []

        # --- Normalize helpers ---
        def norm_label(label: str) -> str:
            label = str(label).lower()
            return label if label in {"positive", "negative", "mixed"} else "mixed"

        # --- Base product signals ---
        base_price = base_product.get("price")
        base_score = base_sentiment.get("avg_sentiment_score", 0.0)
        base_class = base_forecast.get("predicted_class", "")

        # --- Competitor-level insights ---
        for comp in competitors:
            comp_title = comp.get("title", "Competitor")
            comp_price = comp.get("price")
            comp_score = comp.get("avg_sentiment", 0.0)
            comp_class = comp.get("predicted_class", "")

            # Price-performance comparison
            if isinstance(base_price, (int, float)) and isinstance(comp_price, (int, float)):
                if comp_price > base_price and comp_score <= base_score:
                    insights.append(
                        f"{comp_title} is more expensive but does not show stronger customer sentiment than the base product."
                    )
                elif comp_price < base_price and comp_score >= base_score:
                    insights.append(
                        f"{comp_title} may offer better price-performance value with a lower price and comparable sentiment."
                    )

            # Market positioning
            if comp_class != base_class:
                insights.append(
                    f"{comp_title} is positioned in the '{comp_class}' class, while the base product is in the '{base_class}' class."
                )

        # --- Aspect-level insights ---
        sound = norm_label(base_aspect_sentiment.get("sound_quality", {}).get("label"))
        comfort = norm_label(base_aspect_sentiment.get("comfort", {}).get("label"))
        battery = norm_label(base_aspect_sentiment.get("battery_life", {}).get("label"))

        if sound == "negative":
            insights.append("Sound quality appears to be a relative weakness of the base product.")
        if comfort == "positive":
            insights.append("Comfort appears to be a relative strength of the base product.")
        if battery == "positive":
            insights.append("Battery life appears to be a competitive strength of the base product.")

        # --- Meta-insights (pattern detection) ---
        if competitors:
            competitor_prices = [c.get("price") for c in competitors if isinstance(c.get("price"), (int, float))]
            competitor_scores = [c.get("avg_sentiment", 0.0) for c in competitors]

            if competitor_prices:
                if base_price and base_price < min(competitor_prices):
                    insights.append("The base product is the cheapest among similar alternatives.")
                if base_price and base_price > max(competitor_prices):
                    insights.append("The base product is the most expensive option in its competitive set.")

            if competitor_scores:
                if base_score > max(competitor_scores):
                    insights.append("The base product has the strongest customer sentiment among competitors.")
                if base_score < min(competitor_scores):
                    insights.append("The base product has weaker customer sentiment than most competitors.")

        # --- Deduplicate ---
        unique = []
        seen = set()
        for i in insights:
            if i not in seen:
                seen.add(i)
                unique.append(i)

        return unique
