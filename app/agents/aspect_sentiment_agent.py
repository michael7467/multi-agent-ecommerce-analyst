from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.services.aspect_sentiment_service import AspectSentimentService
from app.core.config import settings

class AspectSentimentAgent(BaseAgent):
    def __init__(self, backend: str = "zero_shot") -> None:
        super().__init__(name="AspectSentimentAgent")
        self.service = AspectSentimentService(backend=backend)

    def run(self, product_id: str, top_k: int = 3) -> dict:
        aspect_sentiment = self.service.analyze_product_aspects(
            product_id=product_id,
            top_k=top_k,
        )
        return {"aspect_sentiment": aspect_sentiment}


if __name__ == "__main__":
    agent = AspectSentimentAgent(backend=settings.aspect_sentiment_backend)
    result = agent.run(product_id="B09SPZPDJK", top_k=2)

    print("\n=== ASPECT SENTIMENT ===\n")
    print(result)