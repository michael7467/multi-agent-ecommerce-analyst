from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.services.counterfactual_service import CounterfactualService
from app.observability.agent_tracing import traced_agent

class CounterfactualAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="CounterfactualAgent")
        self.service = CounterfactualService()

    @traced_agent("CounterfactualAgent.run")
    def run(self, product_data: dict) -> dict:

        if not isinstance(product_data, dict):
            raise ValueError("CounterfactualAgent: product_data must be a dict")

        counterfactuals = self.service.generate_counterfactuals(product_data)
        return {"counterfactuals": counterfactuals}