from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.services.competitive_service import CompetitiveService


class CompetitiveAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="CompetitiveAgent")
        self.service = CompetitiveService()

    def run(self, product_id: str, top_k: int = 5) -> dict:
        result = self.service.analyze(product_id=product_id, top_k=top_k)
        return {"competitive_analysis": result}


if __name__ == "__main__":
    agent = CompetitiveAgent()
    result = agent.run(product_id="B09SPZPDJK", top_k=5)

    print("\n=== COMPETITIVE ANALYSIS ===\n")
    print(result)