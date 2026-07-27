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
        # Kept, unlike the equivalent check removed from other agents:
        # CounterfactualService._validate_product_data does
        # "f not in product_data" with no isinstance check of its own, so
        # None or a non-dict here would hit an unclear TypeError inside
        # the service instead of this clean ValueError.
        if not isinstance(product_data, dict):
            raise ValueError("CounterfactualAgent: product_data must be a dict")

        counterfactuals = self.service.generate_counterfactuals(product_data)
        return {"counterfactuals": counterfactuals}