from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.services.counterfactual_service import CounterfactualService


class CounterfactualAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="CounterfactualAgent")
        self.service = CounterfactualService()

    def run(self, product_data: dict) -> dict:
        counterfactuals = self.service.generate_counterfactuals(product_data)
        return {"counterfactuals": counterfactuals}


if __name__ == "__main__":
    from app.agents.data_agent import DataAgent

    data_agent = DataAgent()
    product_data = data_agent.run(product_id="B09SPZPDJK")

    agent = CounterfactualAgent()
    result = agent.run(product_data=product_data)

    print("\n=== COUNTERFACTUALS ===\n")
    for item in result["counterfactuals"]:
        print(item)