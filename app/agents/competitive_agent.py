from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.services.competitive_service import CompetitiveService
from app.observability.agent_tracing import traced_agent

class CompetitiveAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="CompetitiveAgent")
        self.service = CompetitiveService()

    @traced_agent("CompetitiveAgent.run")
    def run(self, product_id: str, top_k: int = 5) -> dict:
        result = self.service.analyze(product_id=product_id, top_k=top_k)
        return {"competitive_analysis": result}