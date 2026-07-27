from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.agents.base_agent import BaseAgent
from app.logging.logger import get_logger
from app.models.llm.llm_client import LLMClient

logger = get_logger("agents.planning_agent")


class PrimaryIntent(str, Enum):
    general = "general"
    sentiment = "sentiment"
    aspect_sentiment = "aspect_sentiment"
    pricing = "pricing"
    comparison = "comparison"
    recommendation = "recommendation"
    visual_similarity = "visual_similarity"
    summarization = "summarization"
    topics = "topics"
    counterfactual = "counterfactual"
    critique = "critique"
    trends = "trends"
    buy_decision = "buy_decision"


class PlanningPlan(BaseModel):
    primary_intent: PrimaryIntent = PrimaryIntent.general
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)

    use_data: bool = True
    use_sentiment: bool = False
    use_aspect_sentiment: bool = False
    use_forecast: bool = False
    use_retrieval: bool = False
    use_recommender: bool = False
    use_image_retrieval: bool = False
    use_summarization: bool = False
    use_topics: bool = False
    use_counterfactuals: bool = False
    use_report: bool = True
    use_guardrail: bool = False
    use_critic: bool = False
    use_competitive: bool = False
    use_buy_decision: bool = False
    use_trends: bool = False

    @model_validator(mode="after")
    def validate_flags(self) -> "PlanningPlan":
        """Runs automatically once after construction / model_validate().

        Does NOT run automatically after later attribute mutation (that
        needs validate_assignment=True on the model config -- which was
        tried here and reverted: this validator itself assigns
        self.use_data/use_guardrail/use_critic, and validate_assignment
        re-invokes this same validator on every assignment it makes,
        including its own, which recurses forever. Simplest fix is calling
        this method explicitly wherever a reconciliation is needed after a
        mutation -- see the end of _rule_boost.
        """
        self.use_data = True
        self.use_guardrail = bool(self.use_forecast and self.use_report)
        if not self.use_report:
            self.use_critic = False
        return self


class PlanningAgent(BaseAgent):
    # Same terms verified behavior-preserving against the original
    # word-boundary-matched rules (see planning_agent.py's _RULES table).
    # Each rule now also carries the PrimaryIntent it corresponds to and a
    # short human-readable reason, both used by _rule_boost below.
    _RULES: list[tuple[list[str], dict[str, bool], PrimaryIntent, str]] = [
        (
            [
                "trend", "trends", "rising categories", "declining categories",
                "emerging complaints", "seasonal", "seasonality",
                "market trend", "category trend",
            ],
            {"use_trends": True, "use_report": True},
            PrimaryIntent.trends,
            "trend query",
        ),
        (
            [
                "should i buy", "should you buy", "buy it", "worth buying",
                "is it worth it", "recommend this product", "would you recommend",
            ],
            {
                "use_buy_decision": True,
                "use_sentiment": True,
                "use_aspect_sentiment": True,
                "use_retrieval": True,
                "use_forecast": True,
                "use_recommender": True,
                "use_report": True,
            },
            PrimaryIntent.buy_decision,
            "buy decision query",
        ),
        (
            [
                "compare", "competitor", "competitors", "vs", "versus",
                "alternative", "alternatives", "tradeoff", "tradeoffs",
                "strengths", "weaknesses", "price performance",
            ],
            {"use_competitive": True, "use_recommender": True, "use_report": True},
            PrimaryIntent.comparison,
            "comparison query",
        ),
        (
            [
                "what if", "counterfactual", "would change",
                "if rating increased", "if reviews increased",
                "how could", "what would need to change",
            ],
            {"use_forecast": True, "use_counterfactuals": True, "use_report": True},
            PrimaryIntent.counterfactual,
            "counterfactual query",
        ),
        (
            [
                "theme", "themes", "topic", "topics",
                "pain point", "pain points", "common issues",
                "common problems", "main problems",
            ],
            {"use_topics": True, "use_report": True},
            PrimaryIntent.topics,
            "topic query",
        ),
        (
            [
                "think", "opinion", "opinions", "feel", "customers", "customer",
                "review", "reviews", "complaint", "complaints", "feedback",
            ],
            {"use_sentiment": True, "use_retrieval": True},
            PrimaryIntent.sentiment,
            "sentiment query",
        ),
        (
            [
                "sound", "sound quality", "battery", "battery life", "comfort",
                "comfortable", "durability", "durable", "build quality",
                "material", "design", "fit", "noise cancellation",
                "noise cancelling", "value", "worth", "price/value",
            ],
            {
                "use_aspect_sentiment": True,
                "use_retrieval": True,
                "use_summarization": True,
            },
            PrimaryIntent.aspect_sentiment,
            "aspect query",
        ),
        (
            [
                "price", "expensive", "cheap", "worth", "overpriced",
                "value", "fair price", "buy", "wait",
            ],
            {
                "use_forecast": True,
                "use_sentiment": True,
                "use_retrieval": True,
                "use_report": True,
            },
            PrimaryIntent.pricing,
            "pricing query",
        ),
        (
            ["similar", "alternative", "alternatives", "recommend", "instead", "compare"],
            {"use_recommender": True},
            PrimaryIntent.recommendation,
            "recommendation query",
        ),
        (
            ["look", "visual", "image", "appearance", "similar-looking"],
            {"use_image_retrieval": True},
            PrimaryIntent.visual_similarity,
            "visual query",
        ),
        (
            ["summarize", "summary", "aspect"],
            {"use_summarization": True, "use_retrieval": True},
            PrimaryIntent.summarization,
            "summarization query",
        ),
        (
            ["evaluate", "verify", "critique", "reliable", "trust", "judge"],
            {"use_critic": True},
            PrimaryIntent.critique,
            "critique query",
        ),
    ]

    def __init__(self) -> None:
        super().__init__(name="PlanningAgent")
        self.llm = LLMClient(model="gpt-4.1-mini")

    def _safe_default_plan(self) -> PlanningPlan:
        return PlanningPlan()

    def _build_prompt(self, query: str) -> str:
        schema = PlanningPlan.model_json_schema()
        schema_json = json.dumps(schema, indent=2)

        return f"""
You are a planning agent in a multi-agent e-commerce intelligence system.

Return ONLY valid JSON matching this schema:
{schema_json}

Rules:
- Always set use_data = true.
- Usually set use_report = true.
- Set use_guardrail = true only when use_forecast and use_report are both true.
- If the query is ambiguous, prefer a smaller set of agents.
- Set primary_intent to the single best intent.
- confidence must be between 0 and 1.
- reasons should be short strings explaining the choice.

User query:
{query}
""".strip()

    def _extract_json(self, raw_response: str) -> dict[str, Any]:
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_response, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    def _normalize_plan(self, plan: dict[str, Any] | PlanningPlan) -> PlanningPlan:
        # validate_flags (above) runs automatically on both model_validate()
        # and on PlanningPlan instances already built -- no manual
        # reconciliation needed here anymore.
        if isinstance(plan, PlanningPlan):
            return plan
        return PlanningPlan.model_validate(plan)

    @staticmethod
    def _matching_terms(text: str, terms: list[str]) -> list[str]:
        return [t for t in terms if re.search(r"\b" + re.escape(t) + r"\b", text)]

    def _rule_boost(self, query: str, plan: PlanningPlan) -> PlanningPlan:
        q = query.lower()

        best_intent: PrimaryIntent | None = None
        best_score = 0

        for terms, flags, intent, reason in self._RULES:
            matched = self._matching_terms(q, terms)
            if not matched:
                continue

            for key, value in flags.items():
                setattr(plan, key, value)

            if reason not in plan.reasons:
                plan.reasons.append(reason)

            # "Primary" intent = whichever rule has the most matching
            # keywords, not whichever rule happens to run last in the loop.
            # Strict > (not >=) means ties keep the earliest-matched intent,
            # so the result is stable and doesn't depend on dict/list
            # reordering elsewhere in the file.
            if len(matched) > best_score:
                best_score = len(matched)
                best_intent = intent

        if best_intent is not None:
            plan.primary_intent = best_intent
            # Confidence now reflects how much explicit keyword evidence
            # backed the winning intent, not just whatever number the LLM
            # guessed before rule_boost potentially overrode half the plan.
            # 1 matching term -> 0.7, 2 -> 0.8, 3 -> 0.9, 4+ -> 1.0.
            plan.confidence = min(1.0, 0.6 + 0.1 * best_score)

        # setattr() above doesn't auto-retrigger validate_flags (see the
        # note on that method), so reconcile once, explicitly, here --
        # rather than re-inlining the same 3 lines a second time.
        return plan.validate_flags()

    def run(self, query: str) -> dict[str, Any]:
        prompt = self._build_prompt(query)

        try:
            raw_response = self.llm.generate_text(prompt)
            parsed = self._extract_json(raw_response)
            plan = self._normalize_plan(parsed)
        except Exception:
            logger.error("PlanningAgent failed to parse or validate LLM response", exc_info=True)
            plan = self._safe_default_plan()

        plan = self._rule_boost(query, plan)
        return {"plan": plan.model_dump()}


if __name__ == "__main__":
    agent = PlanningAgent()

    test_queries = [
        "What do customers think about sound quality?",
        "How is the battery life for this product?",
        "Is this product worth the price?",
        "Show me similar-looking alternatives.",
        "Critique the reliability of this answer.",
    ]

    for query in test_queries:
        result = agent.run(query=query)
        print(f"\nQuery: {query}")
        print(result["plan"])