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
from app.config.settings import settings
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
    agent_execution_seconds,
    agent_errors_total,
    agent_validation_failures_total,
    critic_hallucination_rate,
)
from app.evaluation.metrics.generation_metrics import hallucination_rate_from_critic_scores
from app.agents.buy_decision_agent import BuyDecisionAgent
from app.agents.competitive_agent import CompetitiveAgent
from app.agents.trend_agent import TrendAgent


import traceback
from app.logging.logger import get_logger
from opentelemetry.trace import get_current_span

logger = get_logger("orchestrator")


def _get_trace_id() -> str | None:
    ctx = get_current_span().get_span_context()
    if not ctx or ctx.trace_id == 0:
        return None
    return format(ctx.trace_id, "032x")


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
        self.competitive_agent = CompetitiveAgent()
        self.buy_decision_agent = BuyDecisionAgent()
        self.trend_agent = TrendAgent()
        self.tracer = get_tracer("app.dynamic_orchestrator")

    def _safe_agent_call(
        self,
        name: str,
        fn,
        *,
        critical: bool = True,
        failed_steps: list[str] | None = None,
        **kwargs,
    ):
        """
        Wraps each agent call with:
        - tracing
        - structured logging
        - error isolation
        - graceful fallback for non-critical steps

        Critical steps (critical=True, the default) re-raise on failure and
        abort the whole request -- there's nothing meaningful to return
        without them. Non-critical steps (critical=False) log the failure,
        record the step name in `failed_steps`, and return {} so the rest
        of the pipeline can continue.
        """
        with self.tracer.start_as_current_span(f"{name}.run") as span:
            span.set_attribute("agent", name)
            span.set_attribute("input_args", str(kwargs))
            span.set_attribute("critical", critical)

            try:
                # agent_execution_seconds was defined in metrics.py but
                # never actually recorded anywhere in this project --
                # confirmed by grepping the whole codebase, not assumed.
                # This is the one place that can fix that for every
                # agent at once, since every agent call passes through here.
                with agent_execution_seconds.labels(agent=name).time():
                    result = fn(**kwargs)

                if not isinstance(result, dict):
                    raise ValueError(f"{name} returned non-dict result")

                logger.info(
                    f"{name} completed",
                    extra={
                        "agent": name,
                        "trace_id": _get_trace_id(),
                        "input_args": kwargs
                    },
                )
                return result

            except Exception as exc:
                logger.error(
                    f"{name} failed",
                    extra={
                        "agent": name,
                        "critical": critical,
                        "error": str(exc),
                        "trace_id": _get_trace_id(),
                        "stack": traceback.format_exc(),
                    },
                )
                span.set_attribute("failed", True)
                agent_errors_total.labels(agent=name).inc()

                # ValueError is the convention this codebase uses
                # throughout for input validation failures -- distinct
                # from a genuine dependency/infrastructure failure, and
                # now visible as its own thing instead of folded into the
                # same generic error count as everything else.
                if isinstance(exc, ValueError):
                    agent_validation_failures_total.labels(agent=name).inc()

                if critical:
                    raise

                if failed_steps is not None:
                    failed_steps.append(name)
                return {}

    def run(self, product_id: str, query: str, top_k: int = 3) -> dict:
        ANALYSIS_REQUESTS_TOTAL.labels(endpoint="/analyze").inc()

        with IN_PROGRESS_ANALYSIS.track_inprogress():
            with ANALYSIS_LATENCY_SECONDS.labels(endpoint="/analyze").time():
                with self.tracer.start_as_current_span("dynamic_orchestrator.run") as span:
                    trace_id = _get_trace_id()

                    logger.info(
                        "Starting analysis",
                        extra={
                            "product_id": product_id,
                            "query": query,
                            "top_k": top_k,
                            "trace_id": trace_id,
                        },
                    )

                    try:
                        # -------------------------
                        # Cache lookup
                        # -------------------------
                        cache_payload = {
                            "product_id": product_id,
                            "query": query,
                            "top_k": top_k,
                        }

                        cached = self.cache_service.get_json("analysis:full", cache_payload)
                        if cached:
                            CACHE_HITS_TOTAL.labels(cache_name="analysis_cache").inc()
                            logger.info(
                                "Cache hit",
                                extra={"trace_id": trace_id, "product_id": product_id},
                            )
                            return cached

                        CACHE_MISSES_TOTAL.labels(cache_name="analysis_cache").inc()
                        # -------------------------
                        # Memory + Planning
                        # -------------------------
                        memory = self._safe_agent_call(
                            "memory_agent",
                            self.memory_agent.run,
                            product_id=product_id,
                        )["memory"]

                        plan = self._safe_agent_call(
                            "planning_agent",
                            self.planning_agent.run,
                            query=query,
                        )["plan"]

                        analysis = {
                            "product_id": product_id,
                            "query": query,
                            "memory": memory,
                        }
                        failed_steps: list[str] = []

                        # -------------------------
                        # Execute agents based on plan
                        # -------------------------
                        # 1. Competitive Analysis
                        if plan.get("use_competitive"):
                            competitive = self._safe_agent_call(
                                "competitive_agent",
                                self.competitive_agent.run,
                                critical=False,
                                failed_steps=failed_steps,
                                product_id=product_id,
                                top_k=5,
                            )
                            analysis["competitive_analysis"] = competitive.get("competitive_analysis")
                        # 2. Product Data
                        product_data = {}
                        if plan.get("use_data"):
                            data = self._safe_agent_call(
                                "data_agent",
                                self.data_agent.run,
                                product_id=product_id,
                            )
                            product_data = data
                            analysis.update({
                                "title": data.get("title"),
                                "categories": data.get("categories"),
                                "price": data.get("price"),
                            })

                        # 3. Buy Decision
                        if plan.get("use_buy_decision"):
                            buy_decision = self._safe_agent_call(
                                "buy_decision_agent",
                                self.buy_decision_agent.run,
                                critical=False,
                                failed_steps=failed_steps,
                                analysis_result=analysis,
                            )
                            analysis["buy_decision"] = buy_decision.get("buy_decision")

                        # 4. Topic Modeling
                        if plan.get("use_topics"):
                            topics = self._safe_agent_call(
                                "topic_agent",
                                self.topic_agent.run,
                                critical=False,
                                failed_steps=failed_steps,
                                top_k=5,
                            )
                            analysis["top_themes"] = topics.get("top_themes")
                            analysis["pain_points"] = topics.get("pain_points")
                        # 5. Trend Analysis
                        if plan.get("use_trends"):
                            trends = self._safe_agent_call(
                                "trend_agent",
                                self.trend_agent.run,
                                critical=False,
                                failed_steps=failed_steps,
                            )
                            analysis["trend_analysis"] = trends.get("trend_analysis")

                        # 6. Aspect Sentiment
                        if plan.get("use_aspect_sentiment"):
                            aspect_sentiment = self._safe_agent_call(
                                "aspect_sentiment_agent",
                                self.aspect_sentiment_agent.run,
                                critical=False,
                                failed_steps=failed_steps,
                                product_id=product_id,
                                top_k=2,
                            )
                            analysis["aspect_sentiment"] = aspect_sentiment.get("aspect_sentiment")

                        # 7. Sentiment Analysis
                        if plan.get("use_sentiment"):
                            sentiment = self._safe_agent_call(
                                "sentiment_agent",
                                self.sentiment_agent.run,
                                critical=False,
                                failed_steps=failed_steps,
                                product_id=product_id,
                            )
                            analysis["sentiment"] = sentiment
                        # 8. Forecasting
                        if plan.get("use_forecast"):
                            if not product_data:
                                logger.warning(
                                    "Skipping forecast_agent: forecasting requires product data.",
                                    extra={"agent": "forecast_agent", "trace_id": trace_id},
                                )
                                failed_steps.append("forecast_agent")
                            else:
                                forecast = self._safe_agent_call(
                                    "forecast_agent",
                                    self.forecast_agent.run,
                                    critical=False,
                                    failed_steps=failed_steps,
                                    product_data=product_data,
                                )
                                analysis["predicted_class"] = forecast.get("predicted_class")

                        # 9. Counterfactuals
                        if plan.get("use_counterfactuals"):
                            if not product_data:
                                logger.warning(
                                    "Skipping counterfactual_agent: counterfactual analysis requires product data.",
                                    extra={"agent": "counterfactual_agent", "trace_id": trace_id},
                                )
                                failed_steps.append("counterfactual_agent")
                            else:
                                counterfactuals = self._safe_agent_call(
                                    "counterfactual_agent",
                                    self.counterfactual_agent.run,
                                    critical=False,
                                    failed_steps=failed_steps,
                                    product_data=product_data,
                                )
                                analysis["counterfactuals"] = counterfactuals.get("counterfactuals")

                        # 10. Retrieval (RAG)
                        if plan.get("use_retrieval"):
                            retrieval = self._safe_agent_call(
                                "retrieval_agent",
                                self.retrieval_agent.run,
                                critical=False,
                                failed_steps=failed_steps,
                                product_id=product_id,
                                query=query,
                                top_k=top_k,
                            )
                            analysis["evidence"] = retrieval.get("evidence")
                        # 11. Recommender
                        if plan.get("use_recommender"):
                            recs = self._safe_agent_call(
                                "recommender_agent",
                                self.recommender_agent.run,
                                critical=False,
                                failed_steps=failed_steps,
                                product_id=product_id,
                                top_k=3,
                            )
                            analysis["recommendations"] = recs.get("recommendations")

                        # 12. Image Retrieval
                        if plan.get("use_image_retrieval"):
                            images = self._safe_agent_call(
                                "image_retrieval_agent",
                                self.image_retrieval_agent.run,
                                critical=False,
                                failed_steps=failed_steps,
                                product_id=product_id,
                                top_k=3,
                            )
                            analysis["image_similar_products"] = images.get("image_similar_products")

                        # 13. Summarization
                        if plan.get("use_summarization"):
                            summaries = self._safe_agent_call(
                                "summarization_agent",
                                self.summarization_agent.run,
                                critical=False,
                                failed_steps=failed_steps,
                                product_id=product_id,
                                top_k=2,
                            )
                            analysis["aspect_summaries"] = summaries.get("aspect_summaries")

                        # 14. Report Generation
                        if plan.get("use_report"):
                            with REPORT_LATENCY_SECONDS.labels(model=settings.llm_model).time():
                                report = self._safe_agent_call(
                                    "report_agent",
                                    self.report_agent.run,
                                    critical=False,
                                    failed_steps=failed_steps,
                                    analysis_result=analysis,
                                )
                                if report.get("report"):
                                    analysis["report"] = report["report"]

                        # 15. Guardrail
                        if plan.get("use_guardrail") and "predicted_class" in analysis and "report" in analysis:
                            guardrail = self._safe_agent_call(
                                "guardrail_agent",
                                self.guardrail_agent.run,
                                critical=False,
                                failed_steps=failed_steps,
                                predicted_class=analysis["predicted_class"],
                                report=analysis["report"],
                            )
                            analysis["guardrail_status"] = guardrail.get("status")
                        # 16. Critic Agent
                        if plan.get("use_critic") and "report" in analysis:
                            critic = self._safe_agent_call(
                                "critic_agent",
                                self.critic_agent.run,
                                critical=False,
                                failed_steps=failed_steps,
                                analysis_result=analysis,
                                report=analysis["report"],
                            )
                            analysis["critic_report"] = critic.get("critic_report")
                            analysis["critic_scores"] = critic.get("critic_scores")

                            # Live signal, not the offline FaithfulnessJudge
                            # check -- zero extra cost, reuses the score
                            # critic_agent already computed. Only observed
                            # when the score actually parsed; a None here
                            # means the LLM didn't follow critic_agent's
                            # requested format, not that hallucination is 0.
                            if analysis["critic_scores"]:
                                rate = hallucination_rate_from_critic_scores(analysis["critic_scores"])
                                if rate is not None:
                                    critic_hallucination_rate.observe(rate)

                        analysis["failed_steps"] = failed_steps

                        # -------------------------
                        # Save memory
                        # -------------------------
                        if "report" in analysis:
                            self.memory_agent.save_product_memory(analysis)
                            self.memory_agent.save_history(
                                product_id=product_id,
                                query=query,
                                report=analysis["report"],
                            )

                        final = {"plan": plan, "final_output": analysis}

                        # -------------------------
                        # Cache write
                        # -------------------------
                        # Don't cache a degraded result -- a step that failed
                        # transiently (rate limit, timeout) shouldn't get
                        # served back to every identical request for the next
                        # hour just because it failed once.
                        if not failed_steps:
                            self.cache_service.set_json(
                                "analysis:full",
                                cache_payload,
                                final,
                                ttl_seconds=3600,
                            )

                        logger.info(
                            "Analysis completed",
                            extra={"trace_id": trace_id, "product_id": product_id},
                        )

                        return final

                    except Exception:
                        ANALYSIS_ERRORS_TOTAL.labels(endpoint="/analyze").inc()
                        logger.error(
                            "Analysis failed",
                            extra={
                                "product_id": product_id,
                                "query": query,
                                "trace_id": trace_id,
                                "stack": traceback.format_exc(),
                            },
                        )
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