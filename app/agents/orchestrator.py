from __future__ import annotations

from app.agents.data_agent import DataAgent
from app.agents.sentiment_agent import SentimentAgent
from app.agents.forecast_agent import ForecastAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.recommender_agent import RecommenderAgent
from app.agents.image_retrieval_agent import ImageRetrievalAgent
from app.agents.summarization_agent import SummarizationAgent
from app.agents.report_agent import ReportAgent
from app.agents.guardrail_agent import GuardrailAgent
from app.agents.critic_agent import CriticAgent

class Orchestrator:
    def __init__(self) -> None:
        self.data_agent = DataAgent()
        self.sentiment_agent = SentimentAgent()
        self.forecast_agent = ForecastAgent()
        self.retrieval_agent = RetrievalAgent()
        self.recommender_agent = RecommenderAgent()
        self.image_retrieval_agent = ImageRetrievalAgent()
        self.summarization_agent = SummarizationAgent()
        self.report_agent = ReportAgent()
        self.guardrail_agent = GuardrailAgent()
        self.critic_agent = CriticAgent()

    def run(self, product_id: str, query: str, top_k: int = 3) -> dict:
        product_data = self.data_agent.run(product_id=product_id)

        sentiment_result = self.sentiment_agent.run(product_id=product_id)

        forecast_result = self.forecast_agent.run(product_data=product_data)

        retrieval_result = self.retrieval_agent.run(
            product_id=product_id,
            query=query,
            top_k=top_k,
        )

        recommendation_result = self.recommender_agent.run(
            product_id=product_id,
            top_k=3,
        )

        image_result = self.image_retrieval_agent.run(
            product_id=product_id,
            top_k=3,
        )

        summarization_result = self.summarization_agent.run(
            product_id=product_id,
            top_k=2,
        )

        analysis_result = {
            "product_id": product_id,
            "title": product_data.get("title", ""),
            "categories": product_data.get("categories", ""),
            "price": product_data.get("price", None),
            "predicted_class": forecast_result["predicted_class"],
            "evidence": retrieval_result["evidence"],
            "sentiment": sentiment_result,
            "recommendations": recommendation_result["recommendations"],
            "image_similar_products": image_result["image_similar_products"],
            "aspect_summaries": summarization_result["aspect_summaries"],
        }

        report_result = self.report_agent.run(analysis_result=analysis_result)

        guardrail_result = self.guardrail_agent.run(
            predicted_class=forecast_result["predicted_class"],
            report=report_result["report"],
        )
        critic_result = self.critic_agent.run(
            analysis_result=analysis_result,
            report=report_result["report"],
        )

        return {
            "product_data": product_data,
            "sentiment": sentiment_result,
            "forecast": forecast_result,
            "retrieval": retrieval_result,
            "recommendations": recommendation_result,
            "image_retrieval": image_result,
            "summarization": summarization_result,
            "report": report_result,
            "critic": critic_result,
            "guardrail": guardrail_result,
            "final_output": {
                **analysis_result,
                "report": report_result["report"],
                "guardrail_status": guardrail_result["status"],
                "critic_report": critic_result["critic_report"],
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