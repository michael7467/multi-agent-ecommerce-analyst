from __future__ import annotations

import re
from app.agents.base_agent import BaseAgent
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent

logger = get_logger("agents.guardrail")

class GuardrailAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="GuardrailAgent")

    @traced_agent
    def run(self, predicted_class: str, report: str) -> dict:
        if not isinstance(predicted_class, str):
            raise ValueError("GuardrailAgent: predicted_class must be a string")
        if not isinstance(report, str):
            raise ValueError("GuardrailAgent: report must be a string")

        pattern = r"\b" + re.escape(predicted_class.lower()) + r"\b"
        aligned = bool(re.search(pattern, report.lower()))

        if not aligned:
            logger.info(
                f"{self.name}: alignment failed for predicted_class='{predicted_class}'"
            )

        return {
            "is_aligned": aligned,
            "status": "passed" if aligned else "failed"
        }
