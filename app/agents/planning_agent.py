from __future__ import annotations

import json
import re
from typing import Any

from app.agents.base_agent import BaseAgent
from app.logging.logger import get_logger
from app.models.llm.llm_client import LLMClient
from app.observability.metrics import query_type_total

logger = get_logger("agents.planning_agent")


class PlanningAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="PlanningAgent")
        self.llm = LLMClient(model="gpt-4.1-mini")

    def _safe_default_plan(self) -> dict[str, bool]:
        return {
            "use_data": True,
            "use_sentiment": False,
            "use_aspect_sentiment": False,
            "use_forecast": False,
            "use_retrieval": False,
            "use_recommender": False,
            "use_image_retrieval": False,
            "use_summarization": False,
            "use_topics": False,
            "use_counterfactuals": False,
            "use_report": True,
            "use_guardrail": False,
            "use_critic": False,
            "use_competitive": False,
            "use_buy_decision": False,
            "use_trends": False,
        }

    def _build_prompt(self, query: str) -> str:
        json_schema = json.dumps(self._safe_default_plan(), indent=2)

        return f"""
You are a planning agent in a multi-agent e-commerce intelligence system.

Your task is to read the user's query and decide which agents should be used.

Available agents:
- use_data: load product metadata and core product information
- use_sentiment: analyze overall customer sentiment
- use_aspect_sentiment: analyze sentiment for specific product aspects such as sound quality, battery life, comfort, durability, build quality, or price/value
- use_forecast: predict price class / value positioning
- use_retrieval: retrieve relevant reviews as evidence
- use_recommender: recommend similar products by text/content
- use_image_retrieval: retrieve visually similar products
- use_summarization: summarize reviews by aspect
- use_topics: extract common themes and customer pain points from topic modeling
- use_counterfactuals: generate counterfactual explanations showing how feature changes could alter the model prediction
- use_report: generate a natural language answer
- use_guardrail: check whether the report is aligned with the predicted class
- use_critic: evaluate the quality of the answer
- use_competitive: compare with competitor or alternative products
- use_buy_decision: generate a buy / do-not-buy recommendation
- use_trends: analyze category or market trends

Planning guidance:
- Questions about customer opinion, complaints, quality, or review evidence usually need retrieval and often sentiment.
- Questions about specific aspects like sound quality, battery life, comfort, durability, build quality, or value should usually use aspect sentiment, retrieval, and summarization.
- Questions about price, expensive, cheap, worth it, overpriced, value, or fair price should usually use forecast, sentiment, retrieval, report, and guardrail.
- Questions about similar, alternative, recommend, compare, or instead should usually use recommender.
- Questions about visual appearance, image, look, design, or similar-looking should usually use image retrieval.
- Questions about themes, topics, pain points, common issues, or common problems should usually use topics.
- Questions about what would change the prediction, "what if" scenarios, or how the product could move to another price class should usually use forecast, counterfactuals, report, and guardrail.
- Questions asking to evaluate reliability, verify, critique, or judge quality should use critic.

Rules:
- Always set use_data = true.
- Usually set use_report = true.
- Set use_guardrail = true only if use_forecast and use_report are both true.
- Set use_critic = true only if the user explicitly asks to evaluate, verify, critique, or judge reliability.
- Output valid JSON only.
- Do not include markdown.
- Do not include explanations.

User query:
{query}

Return JSON with exactly these keys and boolean values:
{json_schema}
""".strip()

    def _extract_json(self, raw_response: str) -> dict[str, Any]:
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_response, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    @staticmethod
    def _contains_any(text: str, terms: list[str]) -> bool:
      
        return any(
            re.search(r"\b" + re.escape(term) + r"\b", text) for term in terms
        )

    def _normalize_plan(self, plan: dict[str, Any]) -> dict[str, bool]:
        default = self._safe_default_plan()
        normalized: dict[str, bool] = {}

        for key, default_value in default.items():
            value = plan.get(key, default_value)
            normalized[key] = value if isinstance(value, bool) else default_value

        normalized["use_data"] = True

        normalized["use_guardrail"] = (
            normalized["use_forecast"] and normalized["use_report"]
        )

        if not normalized["use_report"]:
            normalized["use_critic"] = False

        return normalized

 
    _RULES: list[tuple[list[str], dict[str, bool]]] = [
        (
            [
                "trend", "trends", "rising categories", "declining categories",
                "emerging complaints", "seasonal", "seasonality",
                "market trend", "category trend",
            ],
            {"use_trends": True, "use_report": True},
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
        ),
        (
            [
                "compare", "competitor", "competitors", "vs", "versus",
                "alternative", "alternatives", "tradeoff", "tradeoffs",
                "strengths", "weaknesses", "price performance",
            ],
            {"use_competitive": True, "use_recommender": True, "use_report": True},
        ),
        (
            [
                "what if", "counterfactual", "would change",
                "if rating increased", "if reviews increased",
                "how could", "what would need to change",
            ],
            {"use_forecast": True, "use_counterfactuals": True, "use_report": True},
        ),
        (
            [
                "theme", "themes", "topic", "topics",
                "pain point", "pain points", "common issues",
                "common problems", "main problems",
            ],
            {"use_topics": True, "use_report": True},
        ),
        (
            [
                "think", "opinion", "opinions", "feel", "customers", "customer",
                "review", "reviews", "complaint", "complaints", "feedback",
            ],
            {"use_sentiment": True, "use_retrieval": True},
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
        ),
        (
            ["similar", "alternative", "alternatives", "recommend", "instead", "compare"],
            {"use_recommender": True},
        ),
        (
            ["look", "visual", "image", "appearance", "similar-looking"],
            {"use_image_retrieval": True},
        ),
        (
            ["summarize", "summary", "aspect"],
            {"use_summarization": True, "use_retrieval": True},
        ),
        (
            ["evaluate", "verify", "critique", "reliable", "trust", "judge"],
            {"use_critic": True},
        ),
    ]

 
    _RULE_LABELS: list[str] = [
        "trend", "buy_decision", "competitive", "counterfactual", "topics",
        "opinion", "aspect", "pricing", "similar_products", "visual",
        "summarize", "critique",
    ]

    def _rule_boost(self, query: str, plan: dict[str, bool]) -> dict[str, bool]:
        q = query.lower()

        matched_labels = []
        for (terms, flags), label in zip(self._RULES, self._RULE_LABELS):
            if self._contains_any(q, terms):
                plan.update(flags)
                matched_labels.append(label)

        for label in matched_labels:
            query_type_total.labels(query_type=label).inc()
        if not matched_labels:
            query_type_total.labels(query_type="unmatched").inc()

        return self._normalize_plan(plan)

    def run(self, query: str) -> dict[str, dict[str, bool]]:
        prompt = self._build_prompt(query)

        try:
            raw_response = self.llm.generate_text(prompt)
            plan = self._extract_json(raw_response)
        except Exception:
            logger.error("PlanningAgent failed to parse LLM response", exc_info=True)
            plan = self._safe_default_plan()

        plan = self._normalize_plan(plan)
        plan = self._rule_boost(query, plan)

        return {"plan": plan}


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