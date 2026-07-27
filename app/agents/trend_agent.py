from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.services.trend_detection_service import TrendDetectionService
from app.observability.agent_tracing import traced_agent

class TrendAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="TrendAgent")
        self.service = TrendDetectionService()

    @traced_agent("TrendAgent.run")
    def run(self) -> dict:
        trend_result = self.service.analyze()
        return {"trend_analysis": trend_result}