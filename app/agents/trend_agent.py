from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.services.trend_detection_service import TrendDetectionService


class TrendAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="TrendAgent")
        self.service = TrendDetectionService()

    def run(self) -> dict:
        trend_result = self.service.analyze()
        return {"trend_analysis": trend_result}


if __name__ == "__main__":
    agent = TrendAgent()
    result = agent.run()

    print("\n=== TREND ANALYSIS ===\n")
    print(result)