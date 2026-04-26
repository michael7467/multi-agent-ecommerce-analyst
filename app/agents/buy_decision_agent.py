from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.services.buy_decision_service import BuyDecisionService
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent

logger = get_logger("agents.buy_decision")

class BuyDecisionAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="BuyDecisionAgent")
        self.service = BuyDecisionService()

    @traced_agent("BuyDecisionAgent.run")
    def run(self, analysis_result: dict) -> dict:
        if not isinstance(analysis_result, dict):
            raise ValueError("BuyDecisionAgent: analysis_result must be a dict")

        try:
            decision_result = self.service.make_decision(analysis_result)
        except Exception:
            logger.error(f"{self.name}: buy decision failed", exc_info=True)
            raise

        return {"buy_decision": decision_result}
