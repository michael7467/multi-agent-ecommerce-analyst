from __future__ import annotations

import operator
import traceback
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END
from opentelemetry.trace import get_current_span

from app.agents.memory_agent import MemoryAgent
from app.agents.planning_agent import PlanningAgent
from app.agents.data_agent import DataAgent
from app.agents.sentiment_agent import SentimentAgent
from app.agents.topic_agent import TopicAgent
from app.agents.forecast_agent import ForecastAgent
from app.agents.report_agent import ReportAgent
from app.agents.competitive_agent import CompetitiveAgent
from app.agents.buy_decision_agent import BuyDecisionAgent
from app.agents.trend_agent import TrendAgent
from app.agents.aspect_sentiment_agent import AspectSentimentAgent
from app.agents.counterfactual_agent import CounterfactualAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.recommender_agent import RecommenderAgent
from app.agents.image_retrieval_agent import ImageRetrievalAgent
from app.agents.summarization_agent import SummarizationAgent
from app.agents.guardrail_agent import GuardrailAgent
from app.agents.critic_agent import CriticAgent
from app.agents.pii_guardrail_agent import PIIGuardrailAgent
from app.services.cache_service import CacheService
from app.observability.tracing import get_tracer
from app.observability.metrics import (
    agent_execution_seconds,
    agent_errors_total,
    agent_validation_failures_total,
    critic_hallucination_rate,
    ANALYSIS_ERRORS_TOTAL,
    ANALYSIS_LATENCY_SECONDS,
    ANALYSIS_REQUESTS_TOTAL,
    CACHE_HITS_TOTAL,
    CACHE_MISSES_TOTAL,
    IN_PROGRESS_ANALYSIS,
)
from app.evaluation.metrics.generation_metrics import hallucination_rate_from_critic_scores
from app.logging.logger import get_logger

logger = get_logger("langgraph_orchestrator")
tracer = get_tracer("app.langgraph_orchestrator")


def _get_trace_id() -> str | None:
    
    ctx = get_current_span().get_span_context()
    if not ctx or ctx.trace_id == 0:
        return None
    return format(ctx.trace_id, "032x")


class AnalysisState(TypedDict, total=False):


    product_id: str
    query: str
    top_k: int
    memory: dict
    plan: dict
    product_data: dict
    title: str | None
    categories: str | None
    price: float | None
    sentiment: dict | None
    top_themes: list | None
    pain_points: list | None
    predicted_class: str | None
    report: str | None
    competitive_analysis: dict | None
    buy_decision: dict | None
    trend_analysis: dict | None
    aspect_sentiment: dict | None
    counterfactuals: list | None
    evidence: list | None
    recommendations: list | None
    image_similar_products: list | None
    aspect_summaries: dict | None
    guardrail_status: str | None
    critic_report: str | None
    critic_scores: dict | None
    pii_detected: list[str] | None
    failed_steps: Annotated[list[str], operator.add]


def _safe_agent_node(name: str, fn, *, critical: bool = True, **kwargs) -> dict:

    with tracer.start_as_current_span(f"{name}.run") as span:
        span.set_attribute("agent", name)
        span.set_attribute("critical", critical)

        try:
            with agent_execution_seconds.labels(agent=name).time():
                result = fn(**kwargs)

            if not isinstance(result, dict):
                raise ValueError(f"{name} returned non-dict result")

            logger.info(f"{name} completed", extra={"agent": name})
            return result

        except Exception as exc:
            logger.error(
                f"{name} failed",
                extra={
                    "agent": name,
                    "critical": critical,
                    "error": str(exc),
                    "stack": traceback.format_exc(),
                },
            )
            span.set_attribute("failed", True)
            agent_errors_total.labels(agent=name).inc()

            if isinstance(exc, ValueError):
                agent_validation_failures_total.labels(agent=name).inc()

            if critical:
                raise

            return {"failed_steps": [name]}


class LangGraphOrchestrator:
    def __init__(self) -> None:
        self.memory_agent = MemoryAgent()
        self.planning_agent = PlanningAgent()
        self.data_agent = DataAgent()
        self.sentiment_agent = SentimentAgent()
        self.topic_agent = TopicAgent()
        self.forecast_agent = ForecastAgent()
        self.report_agent = ReportAgent()
        self.competitive_agent = CompetitiveAgent()
        self.buy_decision_agent = BuyDecisionAgent()
        self.trend_agent = TrendAgent()

        self.aspect_sentiment_agent = AspectSentimentAgent(backend="zero_shot")
        self.counterfactual_agent = CounterfactualAgent()
        self.retrieval_agent = RetrievalAgent()
        self.recommender_agent = RecommenderAgent()
        self.image_retrieval_agent = ImageRetrievalAgent()
        self.summarization_agent = SummarizationAgent()
        self.guardrail_agent = GuardrailAgent()
        self.critic_agent = CriticAgent()
        self.pii_guardrail_agent = PIIGuardrailAgent()
        self.cache_service = CacheService()
        self.graph = self._build_graph().compile()

    def _memory_node(self, state: AnalysisState) -> dict:
        result = _safe_agent_node(
            "memory_agent", self.memory_agent.run, critical=True, product_id=state["product_id"]
        )
        return {"memory": result["memory"]}

    def _planning_node(self, state: AnalysisState) -> dict:
        result = _safe_agent_node(
            "planning_agent", self.planning_agent.run, critical=True, query=state["query"]
        )
        return {"plan": result["plan"]}


    def _data_node(self, state: AnalysisState) -> dict:
        result = _safe_agent_node(
            "data_agent", self.data_agent.run, critical=False, product_id=state["product_id"]
        )
        if not result:
            return {"failed_steps": result.get("failed_steps", ["data_agent"])}
        return {
            "product_data": result,
            "title": result.get("title"),
            "categories": result.get("categories"),
            "price": result.get("price"),
        }

    def _sentiment_node(self, state: AnalysisState) -> dict:
        result = _safe_agent_node(
            "sentiment_agent", self.sentiment_agent.run, critical=False, product_id=state["product_id"]
        )
        if "failed_steps" in result:
            return result
        return {"sentiment": result}

    def _topic_node(self, state: AnalysisState) -> dict:
        result = _safe_agent_node("topic_agent", self.topic_agent.run, critical=False, top_k=5)
        if "failed_steps" in result:
            return result
        return {"top_themes": result.get("top_themes"), "pain_points": result.get("pain_points")}

    def _competitive_node(self, state: AnalysisState) -> dict:
        result = _safe_agent_node(
            "competitive_agent", self.competitive_agent.run,
            critical=False, product_id=state["product_id"], top_k=5,
        )
        if "failed_steps" in result:
            return result
        return {"competitive_analysis": result.get("competitive_analysis")}

    def _trend_node(self, state: AnalysisState) -> dict:
        result = _safe_agent_node("trend_agent", self.trend_agent.run, critical=False)
        if "failed_steps" in result:
            return result
        return {"trend_analysis": result.get("trend_analysis")}

    def _aspect_sentiment_node(self, state: AnalysisState) -> dict:
  
        result = _safe_agent_node(
            "aspect_sentiment_agent", self.aspect_sentiment_agent.run,
            critical=False, product_id=state["product_id"], top_k=2,
        )
        if "failed_steps" in result:
            return result
        return {"aspect_sentiment": result.get("aspect_sentiment")}

    def _retrieval_node(self, state: AnalysisState) -> dict:
        result = _safe_agent_node(
            "retrieval_agent", self.retrieval_agent.run,
            critical=False, product_id=state["product_id"], query=state["query"], top_k=state["top_k"],
        )
        if "failed_steps" in result:
            return result
        return {"evidence": result.get("evidence")}

    def _recommender_node(self, state: AnalysisState) -> dict:
    
        result = _safe_agent_node(
            "recommender_agent", self.recommender_agent.run,
            critical=False, product_id=state["product_id"], top_k=3,
        )
        if "failed_steps" in result:
            return result
        return {"recommendations": result.get("recommendations")}

    def _image_retrieval_node(self, state: AnalysisState) -> dict:
 
        result = _safe_agent_node(
            "image_retrieval_agent", self.image_retrieval_agent.run,
            critical=False, product_id=state["product_id"], top_k=3,
        )
        if "failed_steps" in result:
            return result
        return {"image_similar_products": result.get("image_similar_products")}

    def _summarization_node(self, state: AnalysisState) -> dict:
    
        result = _safe_agent_node(
            "summarization_agent", self.summarization_agent.run,
            critical=False, product_id=state["product_id"], top_k=2,
        )
        if "failed_steps" in result:
            return result
        return {"aspect_summaries": result.get("aspect_summaries")}

 
    def _forecast_node(self, state: AnalysisState) -> dict:
  
        product_data = state.get("product_data")
        if not product_data:
            logger.warning(
                "Skipping forecast_agent: forecasting requires product data.",
                extra={"agent": "forecast_agent"},
            )
            return {"failed_steps": ["forecast_agent"]}

        result = _safe_agent_node(
            "forecast_agent", self.forecast_agent.run, critical=False, product_data=product_data
        )
        if "failed_steps" in result:
            return result
        return {"predicted_class": result.get("predicted_class")}

    def _counterfactual_node(self, state: AnalysisState) -> dict:
        product_data = state.get("product_data")
        if not product_data:
            logger.warning(
                "Skipping counterfactual_agent: counterfactual analysis requires product data.",
                extra={"agent": "counterfactual_agent"},
            )
            return {"failed_steps": ["counterfactual_agent"]}

        result = _safe_agent_node(
            "counterfactual_agent", self.counterfactual_agent.run, critical=False, product_data=product_data
        )
        if "failed_steps" in result:
            return result
        return {"counterfactuals": result.get("counterfactuals")}

    def _buy_decision_node(self, state: AnalysisState) -> dict:
     
        analysis_result = {
            k: v for k, v in state.items() if k not in ("plan", "failed_steps")
        }
        result = _safe_agent_node(
            "buy_decision_agent", self.buy_decision_agent.run,
            critical=False, analysis_result=analysis_result,
        )
        if "failed_steps" in result:
            return result
        return {"buy_decision": result.get("buy_decision")}


    def _report_node(self, state: AnalysisState) -> dict:
        analysis_result = {
            k: v for k, v in state.items() if k not in ("plan", "failed_steps")
        }
        result = _safe_agent_node(
            "report_agent", self.report_agent.run, critical=False, analysis_result=analysis_result
        )
        if "failed_steps" in result:
            return result
        return {"report": result.get("report")}


    def _guardrail_node(self, state: AnalysisState) -> dict:
 
        plan = state.get("plan", {})
        if not plan.get("use_guardrail") or "predicted_class" not in state or "report" not in state:
            return {}

        result = _safe_agent_node(
            "guardrail_agent", self.guardrail_agent.run, critical=False,
            predicted_class=state["predicted_class"], report=state["report"],
        )
        if "failed_steps" in result:
            return result
        return {"guardrail_status": result.get("status")}

    def _critic_node(self, state: AnalysisState) -> dict:
        plan = state.get("plan", {})
        if not plan.get("use_critic") or "report" not in state:
            return {}

        analysis_result = {
            k: v for k, v in state.items() if k not in ("plan", "failed_steps")
        }
        result = _safe_agent_node(
            "critic_agent", self.critic_agent.run, critical=False,
            analysis_result=analysis_result, report=state["report"],
        )
        if "failed_steps" in result:
            return result

        critic_scores = result.get("critic_scores")

        if critic_scores:
            rate = hallucination_rate_from_critic_scores(critic_scores)
            if rate is not None:
                critic_hallucination_rate.observe(rate)

        return {"critic_report": result.get("critic_report"), "critic_scores": critic_scores}

    def _pii_guardrail_node(self, state: AnalysisState) -> dict:
   
        if "report" not in state:
            return {}

        result = _safe_agent_node(
            "pii_guardrail_agent", self.pii_guardrail_agent.run, critical=False,
            report=state.get("report"), evidence=state.get("evidence"),
        )
        if "failed_steps" in result:
            return result


        return result

    # -------------------------
    # Routing
    # -------------------------
    def _route_after_planning(self, state: AnalysisState) -> list[str]:
        plan = state["plan"]
        targets = []

        if plan.get("use_sentiment"):
            targets.append("sentiment_agent")
        if plan.get("use_topics"):
            targets.append("topic_agent")
        if plan.get("use_competitive"):
            targets.append("competitive_agent")
        if plan.get("use_trends"):
            targets.append("trend_agent")
        if plan.get("use_aspect_sentiment"):
            targets.append("aspect_sentiment_agent")
        if plan.get("use_retrieval"):
            targets.append("retrieval_agent")
        if plan.get("use_recommender"):
            targets.append("recommender_agent")
        if plan.get("use_image_retrieval"):
            targets.append("image_retrieval_agent")
        if plan.get("use_summarization"):
            targets.append("summarization_agent")

        if plan.get("use_data"):
            targets.append("data_agent")
        else:
       
            if plan.get("use_forecast"):
                targets.append("forecast_agent")
            if plan.get("use_counterfactuals"):
                targets.append("counterfactual_agent")
            if plan.get("use_buy_decision"):
                targets.append("buy_decision_agent")

        if not targets:
            targets.append("report_agent")
        return targets

    def _route_after_data(self, state: AnalysisState) -> list[str]:
        plan = state["plan"]
        targets = []
        if plan.get("use_forecast"):
            targets.append("forecast_agent")
        if plan.get("use_counterfactuals"):
            targets.append("counterfactual_agent")
        if plan.get("use_buy_decision"):
            targets.append("buy_decision_agent")
        if not targets:
            targets.append("report_agent")
        return targets

 
    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AnalysisState)

        graph.add_node("memory", self._memory_node)
        graph.add_node("planning", self._planning_node)
        graph.add_node("sentiment_agent", self._sentiment_node)
        graph.add_node("topic_agent", self._topic_node)
        graph.add_node("competitive_agent", self._competitive_node)
        graph.add_node("trend_agent", self._trend_node)
        graph.add_node("aspect_sentiment_agent", self._aspect_sentiment_node)
        graph.add_node("retrieval_agent", self._retrieval_node)
        graph.add_node("recommender_agent", self._recommender_node)
        graph.add_node("image_retrieval_agent", self._image_retrieval_node)
        graph.add_node("summarization_agent", self._summarization_node)
        graph.add_node("data_agent", self._data_node)
        graph.add_node("forecast_agent", self._forecast_node)
        graph.add_node("counterfactual_agent", self._counterfactual_node)
        graph.add_node("buy_decision_agent", self._buy_decision_node)
        graph.add_node("report_agent", self._report_node)
        graph.add_node("guardrail_agent", self._guardrail_node)
        graph.add_node("critic_agent", self._critic_node)
        graph.add_node("pii_guardrail_agent", self._pii_guardrail_node)

        graph.set_entry_point("memory")
        graph.add_edge("memory", "planning")
        graph.add_conditional_edges("planning", self._route_after_planning)
        graph.add_conditional_edges("data_agent", self._route_after_data)

        for node_name in (
            "sentiment_agent", "topic_agent", "competitive_agent", "trend_agent",
            "aspect_sentiment_agent", "retrieval_agent", "recommender_agent",
            "image_retrieval_agent", "summarization_agent",
            "forecast_agent", "counterfactual_agent", "buy_decision_agent",
        ):
            graph.add_edge(node_name, "report_agent")

        graph.add_edge("report_agent", "guardrail_agent")
        graph.add_edge("report_agent", "critic_agent")
        graph.add_edge("report_agent", "pii_guardrail_agent")
        graph.add_edge("guardrail_agent", END)
        graph.add_edge("critic_agent", END)
        graph.add_edge("pii_guardrail_agent", END)

        return graph

    def run(self, product_id: str, query: str, top_k: int = 3) -> dict:
        ANALYSIS_REQUESTS_TOTAL.labels(endpoint="/analyze").inc()

        with IN_PROGRESS_ANALYSIS.track_inprogress():
            with ANALYSIS_LATENCY_SECONDS.labels(endpoint="/analyze").time():
                with tracer.start_as_current_span("langgraph_orchestrator.run"):
                    trace_id = _get_trace_id()

                    logger.info(
                        "Starting analysis",
                        extra={"product_id": product_id, "query": query, "top_k": top_k, "trace_id": trace_id},
                    )

                    try:
                   
                        cache_payload = {"product_id": product_id, "query": query, "top_k": top_k}

                        cached = self.cache_service.get_json("analysis:full", cache_payload)
                        if cached:
                            CACHE_HITS_TOTAL.labels(cache_name="analysis_cache").inc()
                            logger.info("Cache hit", extra={"trace_id": trace_id, "product_id": product_id})
                            return cached

                        CACHE_MISSES_TOTAL.labels(cache_name="analysis_cache").inc()

                    
                        result = self.graph.invoke(
                            {"product_id": product_id, "query": query, "top_k": top_k, "failed_steps": []}
                        )

                    
                        if "report" in result:
                            self.memory_agent.save_product_memory(result)
                            self.memory_agent.save_history(
                                product_id=product_id, query=query, report=result["report"],
                            )

                        final = {"plan": result.get("plan"), "final_output": result}

                     
                        if not result.get("failed_steps"):
                            self.cache_service.set_json(
                                "analysis:full", cache_payload, final, ttl_seconds=3600,
                            )

                        logger.info("Analysis completed", extra={"trace_id": trace_id, "product_id": product_id})
                        return final

                    except Exception:
                        ANALYSIS_ERRORS_TOTAL.labels(endpoint="/analyze").inc()
                        logger.error(
                            "Analysis failed",
                            extra={
                                "product_id": product_id, "query": query,
                                "trace_id": trace_id, "stack": traceback.format_exc(),
                            },
                        )
                        raise


if __name__ == "__main__":
    orchestrator = LangGraphOrchestrator()
    result = orchestrator.run(
        product_id="B001OC5JKY",
        query="What do customers think about sound quality?",
        top_k=3,
    )
    print("\n=== PLAN ===\n")
    print(result["plan"])
    print("\n=== FINAL OUTPUT ===\n")
    print(result["final_output"])