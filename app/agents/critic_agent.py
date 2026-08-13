from __future__ import annotations

import math

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from app.agents.base_agent import BaseAgent
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent

logger = get_logger("agents.critic")


class CritiqueScores(BaseModel):


    explanation_quality: float = Field(description="Score from 1-10 for explanation quality")
    hallucination_risk: float = Field(
        description="Score from 1-10. 10 = very low hallucination risk, 1 = very high hallucination risk"
    )
    retrieval_relevance: float = Field(description="Score from 1-10 for retrieval relevance")
    recommendation_quality: float = Field(description="Score from 1-10 for recommendation quality")
    overall_score: float = Field(description="Overall score from 1-10")
    critique_text: str = Field(description="Short critique explaining the scores")


class CriticAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="CriticAgent")
   
        self.llm = ChatOpenAI(model="gpt-4.1-mini").with_structured_output(CritiqueScores)

    def _safe_float(self, value):
        try:
            result = float(value)
        except Exception:
            return 0.0

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
            """
        return prompt.strip()

    @traced_agent("CriticAgent.run")
    def run(self, analysis_result: dict, report: str) -> dict:
        if not isinstance(analysis_result, dict):
            raise ValueError("CriticAgent: analysis_result must be a dict")
        if not isinstance(report, str):
            raise ValueError("CriticAgent: report must be a string")

        prompt = self._build_prompt(analysis_result, report)
        result: CritiqueScores = self.llm.invoke(prompt)

  
        critic_report_text = (
            f"Explanation Quality: {result.explanation_quality}/10\n"
            f"Hallucination Risk: {result.hallucination_risk}/10\n"
            f"Retrieval Relevance: {result.retrieval_relevance}/10\n"
            f"Recommendation Quality: {result.recommendation_quality}/10\n"
            f"Overall Score: {result.overall_score}/10\n"
            f"Critique: {result.critique_text}"
        )

        return {
            "critic_report": critic_report_text,
            "critic_scores": result.model_dump(),
        }