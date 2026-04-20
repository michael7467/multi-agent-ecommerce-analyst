from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.services.buy_decision_service import BuyDecisionService


class BuyDecisionAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="BuyDecisionAgent")
        self.service = BuyDecisionService()

    def run(self, analysis_result: dict) -> dict:
        decision_result = self.service.make_decision(analysis_result)
        return {"buy_decision": decision_result}


if __name__ == "__main__":
    sample = {
        "title": "Sample Headphones",
        "price": 59.99,
        "predicted_class": "high",
        "sentiment": {
            "avg_sentiment_score": 0.91,
            "positive_review_ratio": 0.95,
        },
        "aspect_sentiment": {
            "sound_quality": {"label": "mixed"},
            "battery_life": {"label": "positive"},
            "comfort": {"label": "positive"},
            "build_quality": {"label": "mixed"},
            "price_value": {"label": "positive"},
        },
        "evidence": [{"x": 1}, {"x": 2}],
        "recommendations": [{"product_id": "abc"}],
    }

    agent = BuyDecisionAgent()
    print(agent.run(sample))