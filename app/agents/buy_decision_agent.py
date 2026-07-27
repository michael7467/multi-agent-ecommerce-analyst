from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.services.buy_decision_service import BuyDecisionService
from app.observability.agent_tracing import traced_agent

class BuyDecisionAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="BuyDecisionAgent")
        self.service = BuyDecisionService()

    @traced_agent("BuyDecisionAgent.run")
    def run(self, analysis_result: dict) -> dict:
        decision_result = self.service.make_decision(analysis_result)
        return {"buy_decision": decision_result}