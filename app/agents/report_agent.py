from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.services.report_service import ReportService


class ReportAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="ReportAgent")
        self.report_service = ReportService()

    def run(self, analysis_result: dict) -> dict:
        report = self.report_service.generate_report(analysis_result)
        return {
            "report": report
        }