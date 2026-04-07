from __future__ import annotations

from app.agents.base_agent import BaseAgent


class GuardrailAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="GuardrailAgent")

    def run(self, predicted_class: str, report: str) -> dict:
        aligned = predicted_class.lower() in report.lower()

        return {
            "is_aligned": aligned,
            "status": "passed" if aligned else "failed"
        }