from __future__ import annotations

from app.agents.data_agent import DataAgent
from app.agents.forecast_agent import ForecastAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.report_agent import ReportAgent
from app.agents.guardrail_agent import GuardrailAgent


class Orchestrator:
    def __init__(self) -> None:
        self.data_agent = DataAgent()
        self.forecast_agent = ForecastAgent()
        self.retrieval_agent = RetrievalAgent()
        self.report_agent = ReportAgent()
        self.guardrail_agent = GuardrailAgent()

    def run(self, product_id: str, query: str, top_k: int = 3) -> dict:
        product_data = self.data_agent.run(product_id=product_id)

        forecast_result = self.forecast_agent.run(product_data=product_data)

        retrieval_result = self.retrieval_agent.run(
            product_id=product_id,
            query=query,
            top_k=top_k,
        )

        analysis_result = {
            "product_id": product_id,
            "title": product_data.get("title", ""),
            "categories": product_data.get("categories", ""),
            "price": product_data.get("price", None),
            "predicted_class": forecast_result["predicted_class"],
            "evidence": retrieval_result["evidence"],
        }

        report_result = self.report_agent.run(analysis_result=analysis_result)

        guardrail_result = self.guardrail_agent.run(
            predicted_class=forecast_result["predicted_class"],
            report=report_result["report"],
        )

        return {
            "product_data": product_data,
            "forecast": forecast_result,
            "retrieval": retrieval_result,
            "report": report_result,
            "guardrail": guardrail_result,
            "final_output": {
                **analysis_result,
                "report": report_result["report"],
                "guardrail_status": guardrail_result["status"],
            },
        }


if __name__ == "__main__":
    orchestrator = Orchestrator()

    result = orchestrator.run(
        product_id="B09SPZPDJK",
        query="sound quality and noise cancellation",
        top_k=3,
    )

    print("\n=== FINAL MULTI-AGENT OUTPUT ===\n")
    print(result["final_output"])