from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.services.aspect_sentiment_service import AspectSentimentService
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent

logger = get_logger("agents.aspect_sentiment")

class AspectSentimentAgent(BaseAgent):
    def __init__(self, backend: str = "zero_shot") -> None:
        super().__init__(name="AspectSentimentAgent")
        self.service = AspectSentimentService(backend=backend)

    @traced_agent
    def run(self, product_id: str, top_k: int = 3) -> dict:
        if not isinstance(product_id, str):
            raise ValueError("AspectSentimentAgent: product_id must be a string")

        try:
            aspect_sentiment = self.service.analyze_product_aspects(
                product_id=product_id,
                top_k=top_k,
            )
        except Exception as e:
            logger.error(f"{self.name}: aspect sentiment failed", exc_info=True)
            raise

        return {"aspect_sentiment": aspect_sentiment}



