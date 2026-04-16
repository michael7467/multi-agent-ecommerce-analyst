from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.models.llm.llm_client import LLMClient


class CriticAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="CriticAgent")
        self.llm = LLMClient(model="gpt-4.1-mini")

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
            evidence_text.append(
                f"""Evidence {i}:
Review title: {ev.get("review_title", "")}
Review text: {ev.get("review_text", "")}
Score: {ev.get("score", 0):.4f}
"""
            )

        recommendation_text = []
        for i, rec in enumerate(recommendations, start=1):
            recommendation_text.append(
                f"""Recommendation {i}:
Product ID: {rec.get("product_id", "")}
Title: {rec.get("title", "")}
Similarity: {rec.get("similarity_score", 0):.4f}
Predicted Class: {rec.get("predicted_class", "")}
"""
            )

        aspect_text = []
        for aspect, payload in aspect_summaries.items():
            aspect_text.append(
                f"{aspect}: {payload.get('summary', '')}"
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

    def run(self, analysis_result: dict, report: str) -> dict:
        prompt = self._build_prompt(analysis_result, report)
        critique = self.llm.generate_text(prompt)

        return {
            "critic_report": critique
        }


if __name__ == "__main__":
    from app.agents.orchestrator import Orchestrator

    orchestrator = Orchestrator()
    result = orchestrator.run(
        product_id="B09SPZPDJK",
        query="sound quality and noise cancellation",
        top_k=3,
    )

    agent = CriticAgent()
    critic_result = agent.run(
        analysis_result=result["final_output"],
        report=result["final_output"]["report"],
    )

    print("\n=== CRITIC REPORT ===\n")
    print(critic_result["critic_report"])