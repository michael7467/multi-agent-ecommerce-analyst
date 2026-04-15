from __future__ import annotations

from app.models.llm.llm_client import LLMClient


class SummarizationService:
    def __init__(self) -> None:
        self.llm = LLMClient(model="gpt-4.1-mini")

    def summarize_aspect(
        self,
        product_id: str,
        aspect: str,
        evidence: list[dict],
    ) -> str:
        if not evidence:
            return f"No strong evidence was found for {aspect.replace('_', ' ')}."

        evidence_text = []
        for i, ev in enumerate(evidence, start=1):
            evidence_text.append(
                f"""Review {i}:
Title: {ev.get("review_title", "")}
Text: {ev.get("review_text", "")}
Score: {ev.get("score", 0):.4f}
"""
            )

        joined_evidence = "\n".join(evidence_text)

        prompt = f"""
You are an expert e-commerce review analyst.

Your task is to summarize customer feedback for one specific aspect of a product.

Rules:
- Focus only on the aspect: {aspect.replace("_", " ")}.
- Use only the provided review evidence.
- Do not invent facts.
- Write 2-4 sentences.
- Mention if feedback is mixed.
- Output plain text only.

Product ID: {product_id}
Aspect: {aspect.replace("_", " ")}

Review Evidence:
{joined_evidence}

Write a concise aspect-based summary.
""".strip()

        return self.llm.generate_text(prompt)


if __name__ == "__main__":
    from app.services.aspect_service import AspectService

    aspect_service = AspectService()
    summarizer = SummarizationService()

    aspect_evidence = aspect_service.get_aspect_evidence("B09SPZPDJK", top_k=2)

    for aspect, evidence in aspect_evidence.items():
        summary = summarizer.summarize_aspect("B09SPZPDJK", aspect, evidence)
        print(f"\n=== {aspect.upper()} SUMMARY ===")
        print(summary)