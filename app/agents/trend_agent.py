from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.services.trend_detection_service import TrendDetectionService
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent

logger = get_logger("agents.trend")

class TrendAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="TrendAgent")
        self.service = TrendDetectionService()

    @traced_agent("TrendAgent.run")
    def run(self) -> dict:
        try:
            trend_result = self.service.analyze()
        except Exception:
            logger.error(f"{self.name}: trend detection failed", exc_info=True)
            raise

        return {"trend_analysis": trend_result}
