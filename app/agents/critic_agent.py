from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.models.llm.llm_client import LLMClient
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent

logger = get_logger("agents.critic")

class CriticAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="CriticAgent")
        self.llm = LLMClient(model="gpt-4.1-mini")

    def _safe_float(self, value):
        try:
            return float(value)
        except Exception:
            return 0.0

    def _build_prompt(self, analysis_result: dict, report: str) -> str:
        predicted_class = analysis_result.get("predicted_class", "")
        title = analysis_result.get("title", "")
        categories = analysis_result.get("categories", "")
        price = analysis_result.get("price", "")
        evidence = analysis_result.get("evidence", [])
        recommendations = analysis_result.get("recommendations", [])
        aspect_summaries = analysis_result.get("aspect_summaries", {})
        sentiment = analysis_result.get("sentiment", {})

        evidence_text = []
        for i, ev in enumerate(evidence, start=1):
            score = self._safe_float(ev.get("score", 0))
            evidence_text.append(
                f"""Evidence {i}:
Review title: {ev.get("review_title", "")}
Review text: {ev.get("review_text", "")}
Score: {score:.4f}
"""
            )

        recommendation_text = []
        for i, rec in enumerate(recommendations, start=1):
            sim = self._safe_float(rec.get("similarity_score", 0))
            recommendation_text.append(
                f"""Recommendation {i}:
Product ID: {rec.get("product_id", "")}
Title: {rec.get("title", "")}
Similarity: {sim:.4f}
Predicted Class: {rec.get("predicted_class", "")}
"""
            )

        aspect_text = [
            f"{aspect}: {payload.get('summary', '')}"
            for aspect, payload in aspect_summaries.items()
        ]

        prompt = f"""
You are a critic agent in a multi-agent e-commerce intelligence system.

Your task is to evaluate the final product analysis report.

Evaluate the following dimensions from 1 to 10:
1. Explanation quality
2. Hallucination risk (10 = very low hallucination risk, 1 = very high hallucination risk)
3. Retrieval relevance
4. Recommendation quality

Rules:
- Use only the provided analysis data.
- Do not invent facts.
- Be strict and realistic.
- If the report claims something unsupported by evidence, lower the hallucination score.
- If recommendations seem weakly related, lower recommendation quality.
- Output plain text only.

Product:
- Title: {title}
- Categories: {categories}
- Price: {price}
- Predicted class: {predicted_class}

Sentiment:
{sentiment}

Aspect summaries:
{"; ".join(aspect_text)}

Evidence:
{"".join(evidence_text)}

Recommendations:
{"".join(recommendation_text)}

Final report:
{report}

Return your answer in this exact format:

Explanation Quality: <score>/10
Hallucination Risk: <score>/10
Retrieval Relevance: <score>/10
Recommendation Quality: <score>/10
Overall Score: <score>/10
Critique: <short critique>
"""
        return prompt.strip()

    @traced_agent
    def run(self, analysis_result: dict, report: str) -> dict:
        if not isinstance(analysis_result, dict):
            raise ValueError("CriticAgent: analysis_result must be a dict")
        if not isinstance(report, str):
            raise ValueError("CriticAgent: report must be a string")

        try:
            prompt = self._build_prompt(analysis_result, report)
            critique = self.llm.generate_text(prompt)
        except Exception:
            logger.error(f"{self.name}: critique generation failed", exc_info=True)
            raise

        return {"critic_report": critique}
