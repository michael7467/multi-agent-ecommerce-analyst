from __future__ import annotations

import math
import re

from app.agents.base_agent import BaseAgent
from app.models.llm.llm_client import LLMClient
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent

logger = get_logger("agents.critic")

# Matches lines like "Explanation Quality: 7/10" from the prompt's requested
# output format. Scores end up here as actual numbers instead of staying
# buried in a text blob nothing ever reads programmatically -- there's a
# full Grafana/Prometheus stack in this project and none of this was ever
# reaching it.
_SCORE_PATTERN = re.compile(
    r"^\s*(Explanation Quality|Hallucination Risk|Retrieval Relevance|"
    r"Recommendation Quality|Overall Score)\s*:\s*(-?\d+(?:\.\d+)?)\s*/\s*10",
    re.IGNORECASE | re.MULTILINE,
)
_CRITIQUE_PATTERN = re.compile(r"^\s*Critique\s*:\s*(.+)", re.IGNORECASE | re.DOTALL | re.MULTILINE)

_SCORE_KEYS = {
    "explanation quality": "explanation_quality",
    "hallucination risk": "hallucination_risk",
    "retrieval relevance": "retrieval_relevance",
    "recommendation quality": "recommendation_quality",
    "overall score": "overall_score",
}


def _parse_critique(text: str) -> dict:
    """Best-effort parse of the critic's expected output format into
    actual numbers. Returns None for any field the LLM didn't produce in
    the expected shape, rather than guessing -- a missing score should
    read as missing, not as a fabricated default that looks real.
    """
    scores: dict[str, float | None] = {v: None for v in _SCORE_KEYS.values()}

    for match in _SCORE_PATTERN.finditer(text):
        label = match.group(1).strip().lower()
        key = _SCORE_KEYS.get(label)
        if key is None:
            continue
        try:
            scores[key] = float(match.group(2))
        except ValueError:
            pass

    critique_match = _CRITIQUE_PATTERN.search(text)
    scores["critique_text"] = critique_match.group(1).strip() if critique_match else None

    return scores


class CriticAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="CriticAgent")
        self.llm = LLMClient(model="gpt-4.1-mini")

    def _safe_float(self, value):
        try:
            result = float(value)
        except Exception:
            return 0.0
        # float(nan) succeeds -- it doesn't raise -- so the try/except
        # above never actually catches a NaN input on its own. Confirmed:
        # _safe_float(float("nan")) returned nan, not 0.0, and formatting
        # it doesn't crash either, it just puts the literal text "nan"
        # into the prompt where a real number belongs.
        return 0.0 if math.isnan(result) else result

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

        sentiment_text = (
            f"Average sentiment score: {self._safe_float(sentiment.get('avg_sentiment_score', 0.0)):.3f}\n"
            f"Positive reviews: {self._safe_float(sentiment.get('positive_review_ratio', 0.0)):.2%}\n"
            f"Neutral reviews: {self._safe_float(sentiment.get('neutral_review_ratio', 0.0)):.2%}\n"
            f"Negative reviews: {self._safe_float(sentiment.get('negative_review_ratio', 0.0)):.2%}"
        )

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
{sentiment_text}

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

    @traced_agent("CriticAgent.run")
    def run(self, analysis_result: dict, report: str) -> dict:
        if not isinstance(analysis_result, dict):
            raise ValueError("CriticAgent: analysis_result must be a dict")
        if not isinstance(report, str):
            raise ValueError("CriticAgent: report must be a string")

        prompt = self._build_prompt(analysis_result, report)
        critique = self.llm.generate_text(prompt)

        return {
            "critic_report": critique,
            "critic_scores": _parse_critique(critique),
        }