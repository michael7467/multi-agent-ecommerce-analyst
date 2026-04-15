from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.services.recommender_service import RecommenderService


class RecommenderAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="RecommenderAgent")
        self.recommender_service = RecommenderService()

    def run(self, product_id: str, top_k: int = 5) -> dict:
        recommendations = self.recommender_service.recommend_similar_products(
            product_id=product_id,
            top_k=top_k,
        )

        return {
            "recommendations": recommendations
        }

if __name__ == "__main__":
    agent = RecommenderAgent()
    result = agent.run(product_id="B09SPZPDJK", top_k=3)

    print("\n=== RECOMMENDER AGENT OUTPUT ===\n")
    print(result)