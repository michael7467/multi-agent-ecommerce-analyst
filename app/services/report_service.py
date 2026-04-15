from __future__ import annotations

from app.models.llm.llm_client import LLMClient


class ReportService:
    def __init__(self) -> None:
        self.llm = LLMClient(model="gpt-4.1-mini")

    def _build_prompt(self, analysis_result: dict) -> str:
        product_id = analysis_result["product_id"]
        title = analysis_result.get("title", "")
        categories = analysis_result.get("categories", "")
        price = analysis_result.get("price", "")
        predicted_class = analysis_result["predicted_class"]
        evidence = analysis_result["evidence"]

        # ✅ Sentiment extraction
        sentiment = analysis_result.get("sentiment", {})
        avg_sentiment_score = sentiment.get("avg_sentiment_score", 0.0)
        positive_review_ratio = sentiment.get("positive_review_ratio", 0.0)
        neutral_review_ratio = sentiment.get("neutral_review_ratio", 0.0)
        negative_review_ratio = sentiment.get("negative_review_ratio", 0.0)

        # Build evidence text
        evidence_text = []
        for i, ev in enumerate(evidence, start=1):
            evidence_text.append(
                f"""Evidence {i}:
Review title: {ev.get("review_title", "")}
Review text: {ev.get("review_text", "")}
Similarity score: {ev.get("score", 0):.4f}
"""
            )

        joined_evidence = "\n".join(evidence_text)

        # ✅ Updated prompt with sentiment
        prompt = f"""
You are an expert e-commerce AI analyst.

Your task is to explain a price-class prediction using structured data, sentiment signals, and retrieved review evidence.

Strict rules:
- The predicted price class is fixed and must not be changed.
- Do not reinterpret, rename, or override the predicted class.
- If the evidence is mixed, explicitly say so, but keep the predicted class unchanged.
- Use only the provided evidence and sentiment summary.
- Do not invent facts.
- Output plain text only.

Product Information:
- Product ID: {product_id}
- Title: {title}
- Categories: {categories}
- Price: {price}
- Predicted price class: {predicted_class}

Sentiment Summary:
- Average sentiment score: {avg_sentiment_score:.3f}
- Positive reviews: {positive_review_ratio:.2%}
- Neutral reviews: {neutral_review_ratio:.2%}
- Negative reviews: {negative_review_ratio:.2%}

Retrieved Evidence:
{joined_evidence}

Write a professional analyst report with:
1. A one-sentence summary clearly stating the predicted class as {predicted_class}
2. A short explanation combining product features and sentiment trends
3. 2–3 bullet points grounded in retrieved review evidence
4. A brief mention of overall customer perception using sentiment statistics
"""

        return prompt.strip()

    def generate_report(self, analysis_result: dict) -> str:
        prompt = self._build_prompt(analysis_result)
        report = self.llm.generate_text(prompt)

        # ✅ Guardrail: enforce predicted class presence
        predicted_class = str(analysis_result["predicted_class"]).lower()

        if predicted_class not in report.lower():
            return (
                f"Predicted price class: {analysis_result['predicted_class']}\n\n"
                f"The model predicts this class based on product features, sentiment signals, "
                f"and retrieved review evidence. However, the generated explanation was not fully aligned, "
                f"so this safe summary is returned instead."
            )

        return report


if __name__ == "__main__":
    from app.services.analysis_service import AnalysisService

    analysis_service = AnalysisService()
    report_service = ReportService()

    result = analysis_service.analyze_product(
        product_id="B09SPZPDJK",
        query="sound quality and noise cancellation",
        top_k=3,
    )

    report = report_service.generate_report(result)

    print("\n=== FINAL LLM REPORT ===\n")
    print(report)