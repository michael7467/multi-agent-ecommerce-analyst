from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.services.recommender_service import RecommenderService
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent

logger = get_logger("agents.recommender")

class RecommenderAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="RecommenderAgent")
        self.recommender_service = RecommenderService()

    @traced_agent("RecommenderAgent.run")
    def run(self, product_id: str, top_k: int = 5) -> dict:
        if not isinstance(product_id, str):
            raise ValueError("RecommenderAgent: product_id must be a string")

        try:
            recommendations = self.recommender_service.recommend_similar_products(
                product_id=product_id,
                top_k=top_k,
            )
        except Exception:
            logger.error(f"{self.name}: recommendation failed", exc_info=True)
            raise

        if not recommendations:
            return {"recommendations": []}

        return {"recommendations": recommendations}
