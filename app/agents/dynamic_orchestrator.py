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
from app.services.cache_service import CacheService
from app.observability.tracing import get_tracer
from app.observability.metrics import (
    ANALYSIS_ERRORS_TOTAL,
    ANALYSIS_LATENCY_SECONDS,
    ANALYSIS_REQUESTS_TOTAL,
    CACHE_HITS_TOTAL,
    CACHE_MISSES_TOTAL,
    IN_PROGRESS_ANALYSIS,
    REPORT_LATENCY_SECONDS,
)
from app.agents.buy_decision_agent import BuyDecisionAgent
from app.agents.competitive_agent import CompetitiveAgent
from app.agents.trend_agent import TrendAgent

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
        self.cache_service = CacheService()
        self.tracer = get_tracer("app.dynamic_orchestrator")
        self.competitive_agent = CompetitiveAgent()
        self.buy_decision_agent = BuyDecisionAgent()
        self.trend_agent = TrendAgent()

    def run(self, product_id: str, query: str, top_k: int = 3) -> dict:
        ANALYSIS_REQUESTS_TOTAL.inc()

        with IN_PROGRESS_ANALYSIS.track_inprogress():
            with ANALYSIS_LATENCY_SECONDS.time():
                with self.tracer.start_as_current_span("dynamic_orchestrator.run") as span:
                    span.set_attribute("product_id", str(product_id))
                    span.set_attribute("query", str(query))
                    span.set_attribute("top_k", int(top_k))

                    cache_payload = {
                        "product_id": product_id,
                        "query": query,
                        "top_k": top_k,
                    }

                    with self.tracer.start_as_current_span("cache.lookup") as cache_span:
                        cached_result = self.cache_service.get_json("analysis:full", cache_payload)
                        cache_hit = cached_result is not None
                        cache_span.set_attribute("cache.hit", cache_hit)

                    if cached_result is not None:
                        CACHE_HITS_TOTAL.inc()
                        span.set_attribute("cache.hit", True)
                        return cached_result

                    CACHE_MISSES_TOTAL.inc()
                    span.set_attribute("cache.hit", False)

                    try:
                        with self.tracer.start_as_current_span("memory.load"):
                            memory_result = self.memory_agent.run(product_id=product_id)
                            memory = memory_result["memory"]

                        with self.tracer.start_as_current_span("planning.run"):
                            planning_result = self.planning_agent.run(query=query)
                            plan = planning_result["plan"]

                        analysis_result: dict = {
                            "product_id": product_id,
                            "query": query,
                            "memory": memory,
                        }

                        product_data = {}
                        if plan.get("use_competitive", False):
                            with self.tracer.start_as_current_span("competitive_agent.run"):
                                competitive_result = self.competitive_agent.run(
                                    product_id=product_id,
                                    top_k=5,
                                )
                                analysis_result["competitive_analysis"] = competitive_result["competitive_analysis"]
                        if plan.get("use_data", False):
                            with self.tracer.start_as_current_span("data_agent.run"):
                                product_data = self.data_agent.run(product_id=product_id)
                                analysis_result.update(
                                    {
                                        "title": product_data.get("title", ""),
                                        "categories": product_data.get("categories", ""),
                                        "price": product_data.get("price", None),
                                    }
                                )
                        if plan.get("use_buy_decision", False):
                            with self.tracer.start_as_current_span("buy_decision_agent.run"):
                                buy_decision_result = self.buy_decision_agent.run(
                                    analysis_result=analysis_result
                                )
                                analysis_result["buy_decision"] = buy_decision_result["buy_decision"]
                        if plan.get("use_topics", False):
                            with self.tracer.start_as_current_span("topic_agent.run"):
                                topic_result = self.topic_agent.run(top_k=5)
                                analysis_result["top_themes"] = topic_result["top_themes"]
                                analysis_result["pain_points"] = topic_result["pain_points"]

                        if plan.get("use_trends", False):
                            with self.tracer.start_as_current_span("trend_agent.run"):
                                trend_result = self.trend_agent.run()
                                analysis_result["trend_analysis"] = trend_result["trend_analysis"]
                                
                        if plan.get("use_aspect_sentiment", False):
                            with self.tracer.start_as_current_span("aspect_sentiment_agent.run"):
                                aspect_sentiment_result = self.aspect_sentiment_agent.run(
                                    product_id=product_id,
                                    top_k=2,
                                )
                                analysis_result["aspect_sentiment"] = (
                                    aspect_sentiment_result["aspect_sentiment"]
                                )

                        if plan.get("use_sentiment", False):
                            with self.tracer.start_as_current_span("sentiment_agent.run"):
                                sentiment_result = self.sentiment_agent.run(product_id=product_id)
                                analysis_result["sentiment"] = sentiment_result

                        if plan.get("use_forecast", False):
                            with self.tracer.start_as_current_span("forecast_agent.run"):
                                forecast_result = self.forecast_agent.run(product_data=product_data)
                                analysis_result["predicted_class"] = forecast_result["predicted_class"]

                        if plan.get("use_counterfactuals", False):
                            with self.tracer.start_as_current_span("counterfactual_agent.run"):
                                if not product_data:
                                    raise ValueError(
                                        "Counterfactual analysis requires product data, but no product data was loaded."
                                    )
                                counterfactual_result = self.counterfactual_agent.run(
                                    product_data=product_data
                                )
                                analysis_result["counterfactuals"] = (
                                    counterfactual_result["counterfactuals"]
                                )

                        if plan.get("use_retrieval", False):
                            with self.tracer.start_as_current_span("retrieval_agent.run"):
                                retrieval_result = self.retrieval_agent.run(
                                    product_id=product_id,
                                    query=query,
                                    top_k=top_k,
                                )
                                analysis_result["evidence"] = retrieval_result["evidence"]

                        if plan.get("use_recommender", False):
                            with self.tracer.start_as_current_span("recommender_agent.run"):
                                recommendation_result = self.recommender_agent.run(
                                    product_id=product_id,
                                    top_k=3,
                                )
                                analysis_result["recommendations"] = (
                                    recommendation_result["recommendations"]
                                )

                        if plan.get("use_image_retrieval", False):
                            with self.tracer.start_as_current_span("image_retrieval_agent.run"):
                                image_result = self.image_retrieval_agent.run(
                                    product_id=product_id,
                                    top_k=3,
                                )
                                analysis_result["image_similar_products"] = (
                                    image_result["image_similar_products"]
                                )

                        if plan.get("use_summarization", False):
                            with self.tracer.start_as_current_span("summarization_agent.run"):
                                summarization_result = self.summarization_agent.run(
                                    product_id=product_id,
                                    top_k=2,
                                )
                                analysis_result["aspect_summaries"] = (
                                    summarization_result["aspect_summaries"]
                                )

                        if plan.get("use_report", False):
                            with REPORT_LATENCY_SECONDS.time():
                                with self.tracer.start_as_current_span("report_agent.run"):
                                    report_result = self.report_agent.run(
                                        analysis_result=analysis_result
                                    )
                                    analysis_result["report"] = report_result["report"]

                        if (
                            plan.get("use_guardrail", False)
                            and "predicted_class" in analysis_result
                            and "report" in analysis_result
                        ):
                            with self.tracer.start_as_current_span("guardrail_agent.run"):
                                guardrail_result = self.guardrail_agent.run(
                                    predicted_class=analysis_result["predicted_class"],
                                    report=analysis_result["report"],
                                )
                                analysis_result["guardrail_status"] = guardrail_result["status"]

                        if plan.get("use_critic", False) and "report" in analysis_result:
                            with self.tracer.start_as_current_span("critic_agent.run"):
                                critic_result = self.critic_agent.run(
                                    analysis_result=analysis_result,
                                    report=analysis_result["report"],
                                )
                                analysis_result["critic_report"] = critic_result["critic_report"]

                        if "report" in analysis_result:
                            with self.tracer.start_as_current_span("memory.save"):
                                self.memory_agent.save_product_memory(analysis_result)
                                self.memory_agent.save_history(
                                    product_id=product_id,
                                    query=query,
                                    report=analysis_result["report"],
                                )

                        final_result = {
                            "plan": plan,
                            "final_output": analysis_result,
                        }

                        with self.tracer.start_as_current_span("cache.write"):
                            self.cache_service.set_json(
                                "analysis:full",
                                cache_payload,
                                final_result,
                                ttl_seconds=3600,
                            )

                        return final_result

                    except Exception:
                        ANALYSIS_ERRORS_TOTAL.inc()
                        raise


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