from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent
from app.services.class_alignment import check_class_alignment

logger = get_logger("agents.guardrail")


class GuardrailAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="GuardrailAgent")

    @traced_agent("GuardrailAgent.run")
    def run(self, predicted_class: str, report: str) -> dict:
        if not isinstance(predicted_class, str):
            raise ValueError("GuardrailAgent: predicted_class must be a string")
        if not isinstance(report, str):
            raise ValueError("GuardrailAgent: report must be a string")

        result = check_class_alignment(predicted_class, report)

        if result["status"] != "passed":
            logger.info(
                f"{self.name}: alignment check '{result['status']}' for "
                f"predicted_class='{predicted_class}'",
                extra={"reasons": result["reasons"]},
            )

        return result