from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.services.competitive_service import CompetitiveService
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent

logger = get_logger("agents.competitive")

class CompetitiveAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="CompetitiveAgent")
        self.service = CompetitiveService()

    @traced_agent("CompetitiveAgent.run")
    def run(self, product_id: str, top_k: int = 5) -> dict:
        if not isinstance(product_id, str):
            raise ValueError("CompetitiveAgent: product_id must be a string")

        try:
            result = self.service.analyze(product_id=product_id, top_k=top_k)
        except Exception:
            logger.error(f"{self.name}: competitive analysis failed", exc_info=True)
            raise

        return {"competitive_analysis": result}
