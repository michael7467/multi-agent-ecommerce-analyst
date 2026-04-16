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
from app.agents.planning_agent import PlanningAgent
from app.agents.memory_agent import MemoryAgent
from app.agents.aspect_sentiment_agent import AspectSentimentAgent
from app.agents.topic_agent import TopicAgent
from app.agents.counterfactual_agent import CounterfactualAgent


class DynamicOrchestrator:
    def __init__(self) -> None:
        self.planning_agent = PlanningAgent()
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
        self.memory_agent = MemoryAgent()
        self.aspect_sentiment_agent = AspectSentimentAgent(backend="zero_shot")
        self.topic_agent = TopicAgent()
        self.counterfactual_agent = CounterfactualAgent()

    def run(self, product_id: str, query: str, top_k: int = 3) -> dict:
        # 1. Load long-term memory first
        memory_result = self.memory_agent.run(product_id=product_id)
        memory = memory_result["memory"]

        # 2. Let the planner decide which agents to use
        planning_result = self.planning_agent.run(query=query)
        plan = planning_result["plan"]

        # 3. Start building analysis result
        analysis_result: dict = {
            "product_id": product_id,
            "query": query,
            "memory": memory,
        }

        product_data = {}

        # 4. Load product metadata and ML features first
        if plan.get("use_data", False):
            product_data = self.data_agent.run(product_id=product_id)
            analysis_result.update(
                {
                    "title": product_data.get("title", ""),
                    "categories": product_data.get("categories", ""),
                    "price": product_data.get("price", None),
                }
            )

        # 5. Topic modeling / theme extraction
        if "theme" in query.lower() or "topic" in query.lower() or "pain point" in query.lower():
            topic_result = self.topic_agent.run(top_k=5)
            analysis_result["top_themes"] = topic_result["top_themes"]
            analysis_result["pain_points"] = topic_result["pain_points"]

        # 6. Aspect-based sentiment
        if plan.get("use_aspect_sentiment", False):
            aspect_sentiment_result = self.aspect_sentiment_agent.run(
                product_id=product_id,
                top_k=2,
            )
            analysis_result["aspect_sentiment"] = aspect_sentiment_result["aspect_sentiment"]

        # 7. Overall sentiment analysis
        if plan.get("use_sentiment", False):
            sentiment_result = self.sentiment_agent.run(product_id=product_id)
            analysis_result["sentiment"] = sentiment_result

        # 8. Forecast / price-class prediction
        if plan.get("use_forecast", False):
            forecast_result = self.forecast_agent.run(product_data=product_data)
            analysis_result["predicted_class"] = forecast_result["predicted_class"]

        # 9. Counterfactual explanations
        if plan.get("use_counterfactuals", False):
            if not product_data:
                raise ValueError(
                    "Counterfactual analysis requires product data, but no product data was loaded."
                )

            counterfactual_result = self.counterfactual_agent.run(product_data=product_data)
            analysis_result["counterfactuals"] = counterfactual_result["counterfactuals"]

        # 10. Retrieve review evidence
        if plan.get("use_retrieval", False):
            retrieval_result = self.retrieval_agent.run(
                product_id=product_id,
                query=query,
                top_k=top_k,
            )
            analysis_result["evidence"] = retrieval_result["evidence"]

        # 11. Text/content-based recommendations
        if plan.get("use_recommender", False):
            recommendation_result = self.recommender_agent.run(
                product_id=product_id,
                top_k=3,
            )
            analysis_result["recommendations"] = recommendation_result["recommendations"]

        # 12. Image-based similar products
        if plan.get("use_image_retrieval", False):
            image_result = self.image_retrieval_agent.run(
                product_id=product_id,
                top_k=3,
            )
            analysis_result["image_similar_products"] = image_result["image_similar_products"]

        # 13. Aspect-based summarization
        if plan.get("use_summarization", False):
            summarization_result = self.summarization_agent.run(
                product_id=product_id,
                top_k=2,
            )
            analysis_result["aspect_summaries"] = summarization_result["aspect_summaries"]

        # 14. Natural-language answer/report
        if plan.get("use_report", False):
            report_result = self.report_agent.run(analysis_result=analysis_result)
            analysis_result["report"] = report_result["report"]

        # 15. Guardrail only makes sense if forecast + report exist
        if (
            plan.get("use_guardrail", False)
            and "predicted_class" in analysis_result
            and "report" in analysis_result
        ):
            guardrail_result = self.guardrail_agent.run(
                predicted_class=analysis_result["predicted_class"],
                report=analysis_result["report"],
            )
            analysis_result["guardrail_status"] = guardrail_result["status"]

        # 16. Critic evaluates final answer if requested
        if plan.get("use_critic", False) and "report" in analysis_result:
            critic_result = self.critic_agent.run(
                analysis_result=analysis_result,
                report=analysis_result["report"],
            )
            analysis_result["critic_report"] = critic_result["critic_report"]

        # 17. Save to long-term memory only if a report was generated
        if "report" in analysis_result:
            self.memory_agent.save_product_memory(analysis_result)
            self.memory_agent.save_history(
                product_id=product_id,
                query=query,
                report=analysis_result["report"],
            )

        # 18. Return both the plan and the final output
        return {
            "plan": plan,
            "final_output": analysis_result,
        }


if __name__ == "__main__":
    orchestrator = DynamicOrchestrator()

    result = orchestrator.run(
        product_id="B09SPZPDJK",
        query="What do customers think about sound quality and are there similar-looking alternatives?",
        top_k=3,
    )

    print("\n=== PLAN ===\n")
    print(result["plan"])

    print("\n=== FINAL OUTPUT ===\n")
    print(result["final_output"])