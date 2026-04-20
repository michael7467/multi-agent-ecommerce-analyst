from __future__ import annotations

from app.agents.data_agent import DataAgent
from app.agents.sentiment_agent import SentimentAgent
from app.agents.aspect_sentiment_agent import AspectSentimentAgent
from app.agents.forecast_agent import ForecastAgent
from app.agents.recommender_agent import RecommenderAgent


class CompetitiveService:
    def __init__(self) -> None:
        self.data_agent = DataAgent()
        self.sentiment_agent = SentimentAgent()
        self.aspect_sentiment_agent = AspectSentimentAgent(backend="zero_shot")
        self.forecast_agent = ForecastAgent()
        self.recommender_agent = RecommenderAgent()

    def analyze(self, product_id: str, top_k: int = 5) -> dict:
        base_product = self.data_agent.run(product_id=product_id)
        base_sentiment = self.sentiment_agent.run(product_id=product_id)
        base_aspect_sentiment = self.aspect_sentiment_agent.run(
            product_id=product_id,
            top_k=2,
        )["aspect_sentiment"]
        base_forecast = self.forecast_agent.run(product_data=base_product)

        recommendations = self.recommender_agent.run(
            product_id=product_id,
            top_k=top_k,
        )["recommendations"]

        competitors = []

        for rec in recommendations:
            competitor_id = rec["product_id"]

            try:
                comp_data = self.data_agent.run(product_id=competitor_id)
                comp_sentiment = self.sentiment_agent.run(product_id=competitor_id)
                comp_aspect_sentiment = self.aspect_sentiment_agent.run(
                    product_id=competitor_id,
                    top_k=2,
                )["aspect_sentiment"]
                comp_forecast = self.forecast_agent.run(product_data=comp_data)

                competitors.append(
                    {
                        "product_id": competitor_id,
                        "title": comp_data.get("title", ""),
                        "price": comp_data.get("price"),
                        "predicted_class": comp_forecast.get("predicted_class"),
                        "avg_sentiment": comp_sentiment.get("avg_sentiment_score"),
                        "similarity_score": rec.get("similarity_score"),
                        "aspect_sentiment": comp_aspect_sentiment,
                    }
                )
            except Exception:
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

        base_price = base_product.get("price")
        base_score = base_sentiment.get("avg_sentiment_score", 0.0)
        base_class = base_forecast.get("predicted_class", "")

        for comp in competitors:
            comp_title = comp.get("title", "Competitor")
            comp_price = comp.get("price")
            comp_score = comp.get("avg_sentiment", 0.0)
            comp_class = comp.get("predicted_class", "")

            if isinstance(base_price, (int, float)) and isinstance(comp_price, (int, float)):
                if comp_price > base_price and comp_score <= base_score:
                    insights.append(
                        f"{comp_title} is more expensive but does not show stronger customer sentiment than the base product."
                    )
                elif comp_price < base_price and comp_score >= base_score:
                    insights.append(
                        f"{comp_title} may offer better price-performance value with a lower price and comparable sentiment."
                    )

            if comp_class != base_class:
                insights.append(
                    f"{comp_title} is positioned in the '{comp_class}' class, while the base product is in the '{base_class}' class."
                )

        sound = base_aspect_sentiment.get("sound_quality", {})
        comfort = base_aspect_sentiment.get("comfort", {})
        battery = base_aspect_sentiment.get("battery_life", {})

        if sound.get("label") == "negative":
            insights.append("Sound quality appears to be a relative weakness of the base product.")
        if comfort.get("label") == "positive":
            insights.append("Comfort appears to be a relative strength of the base product.")
        if battery.get("label") == "positive":
            insights.append("Battery life appears to be a competitive strength of the base product.")

        return insights