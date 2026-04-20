from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.services.report_service import ReportService
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent

logger = get_logger("agents.report_agent")

class ReportAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="ReportAgent")
        self.report_service = ReportService()

    @traced_agent
    def run(self, analysis_result: dict) -> dict:
        if not isinstance(analysis_result, dict):
            raise ValueError("ReportAgent: analysis_result must be a dict")

        try:
            report = self.report_service.generate_report(analysis_result)
        except Exception as e:
            logger.error(f"{self.name}: report generation failed", exc_info=True)
            raise

        return {"report": report}
