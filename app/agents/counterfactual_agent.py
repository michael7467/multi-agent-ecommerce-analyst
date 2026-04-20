from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.services.counterfactual_service import CounterfactualService
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent

logger = get_logger("agents.counterfactual")

class CounterfactualAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="CounterfactualAgent")
        self.service = CounterfactualService()

    @traced_agent
    def run(self, product_data: dict) -> dict:
        if not isinstance(product_data, dict):
            raise ValueError("CounterfactualAgent: product_data must be a dict")

        try:
            counterfactuals = self.service.generate_counterfactuals(product_data)
        except Exception:
            logger.error(f"{self.name}: counterfactual generation failed", exc_info=True)
            raise

        return {"counterfactuals": counterfactuals}
