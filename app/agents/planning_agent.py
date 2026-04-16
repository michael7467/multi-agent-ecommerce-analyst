from __future__ import annotations

import json

from app.agents.base_agent import BaseAgent
from app.models.llm.llm_client import LLMClient


class PlanningAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="PlanningAgent")
        self.llm = LLMClient(model="gpt-4.1-mini")

  

    def _build_prompt(self, query: str) -> str:
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
    - Always set use_data = true
    - Usually set use_report = true
    - Set use_guardrail = true only if use_forecast and use_report are both true
    - Set use_critic = true only if the user explicitly asks to evaluate, verify, critique, or judge reliability
    - Output valid JSON only
    - Do not include markdown
    - Do not include explanations

    User query:
    {query}

    Return JSON with exactly these keys:
    {{
    "use_data": true,
    "use_sentiment": false,
    "use_aspect_sentiment": false,
    "use_forecast": false,
    "use_retrieval": false,
    "use_recommender": false,
    "use_image_retrieval": false,
    "use_summarization": false,
    "use_topics": false,
    "use_counterfactuals": false,
    "use_report": true,
    "use_guardrail": false,
    "use_critic": false
    }}
    """.strip()
    def _safe_default_plan(self) -> dict:
        return {
            "use_data": True,
            "use_sentiment": False,
            "use_aspect_sentiment": False,
            "use_forecast": False,
            "use_retrieval": False,
            "use_recommender": False,
            "use_image_retrieval": False,
            "use_summarization": False,
            "use_report": True,
            "use_guardrail": False,
            "use_critic": False,
            "use_topics": False,
            "use_counterfactuals": False,
        }

    def _normalize_plan(self, plan: dict) -> dict:
        default = self._safe_default_plan()
        normalized = {}

        for key, default_value in default.items():
            normalized[key] = bool(plan.get(key, default_value))

        normalized["use_data"] = True

        if normalized["use_forecast"] and normalized["use_report"]:
            normalized["use_guardrail"] = True
        else:
            normalized["use_guardrail"] = False

        if not normalized["use_report"]:
            normalized["use_critic"] = False

        return normalized

    def _rule_boost(self, query: str, plan: dict) -> dict:
        q = query.lower()

        aspect_terms = [
            "sound", "sound quality", "battery", "battery life", "comfort",
            "comfortable", "durability", "durable", "build quality",
            "material", "design", "fit", "noise cancellation",
            "noise cancelling", "value", "worth", "price/value"
        ]

        opinion_terms = [
            "think", "opinion", "opinions", "feel", "customers", "customer",
            "review", "reviews", "complaint", "complaints", "feedback"
        ]

        pricing_terms = [
            "price", "expensive", "cheap", "worth", "overpriced",
            "value", "fair price", "buy", "wait"
        ]

        recommendation_terms = [
            "similar", "alternative", "alternatives", "recommend",
            "instead", "compare"
        ]

        visual_terms = [
            "look", "visual", "image", "appearance", "similar-looking"
        ]

        summary_terms = [
            "summarize", "summary", "aspect"
        ]

        critic_terms = [
            "evaluate", "verify", "critique", "reliable", "trust", "judge"
        ]
        topic_terms = [
            "theme", "themes", "topic", "topics",
            "pain point", "pain points", "common issues",
            "common problems", "main problems"
        ]


        counterfactual_terms = [
                "what if",
                "counterfactual",
                "would change",
                "if rating increased",
                "if reviews increased",
                "how could",
                "what would need to change",
            ]

        if any(term in q for term in counterfactual_terms):
            plan["use_data"] = True
            plan["use_forecast"] = True
            plan["use_counterfactuals"] = True
            plan["use_report"] = True

        if any(term in q for term in topic_terms):
            plan["use_topics"] = True
            plan["use_report"] = True
            
        if any(term in q for term in opinion_terms):
            plan["use_sentiment"] = True
            plan["use_retrieval"] = True

        if any(term in q for term in aspect_terms):
            plan["use_aspect_sentiment"] = True
            plan["use_retrieval"] = True
            plan["use_summarization"] = True

        if any(term in q for term in pricing_terms):
            plan["use_forecast"] = True
            plan["use_sentiment"] = True
            plan["use_retrieval"] = True
            plan["use_report"] = True

        if any(term in q for term in recommendation_terms):
            plan["use_recommender"] = True

        if any(term in q for term in visual_terms):
            plan["use_image_retrieval"] = True

        if any(term in q for term in summary_terms):
            plan["use_summarization"] = True
            plan["use_retrieval"] = True

        if any(term in q for term in critic_terms):
            plan["use_critic"] = True

        return self._normalize_plan(plan)

    def run(self, query: str) -> dict:
        prompt = self._build_prompt(query)

        try:
            raw_response = self.llm.generate_text(prompt)
            plan = json.loads(raw_response)
            plan = self._normalize_plan(plan)
        except Exception:
            plan = self._safe_default_plan()

        plan = self._rule_boost(query, plan)

        return {"plan": plan}


if __name__ == "__main__":
    agent = PlanningAgent()

    test_queries = [
        "What do customers think about sound quality?",
        "How is the battery life for this product?",
        "Is this product worth the price?",
        "Show me similar-looking alternatives.",
        "Critique the reliability of this answer."
    ]

    for query in test_queries:
        result = agent.run(query=query)
        print(f"\nQuery: {query}")
        print(result["plan"])