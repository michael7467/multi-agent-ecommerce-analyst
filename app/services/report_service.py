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
        prompt = f"""
        You are an e-commerce AI analyst.

        Your task is to explain a price-class prediction using only the provided evidence.

        Strict rules:
        - The predicted price class is fixed and must not be changed.
        - Do not reinterpret, rename, or override the predicted class.
        - If the evidence is mixed, say the evidence is mixed, but keep the predicted class unchanged.
        - Use only the provided evidence.
        - Do not invent facts.
        - Output plain text only.

        Product ID: {product_id}
        Title: {title}
        Categories: {categories}
        Price: {price}
        Predicted price class: {predicted_class}

        Retrieved evidence:
        {joined_evidence}

        Write a short analyst report with:
        1. A one-sentence summary that explicitly states the predicted class as {predicted_class}
        2. A brief explanation of why this prediction may make sense
        3. 2-3 bullet points grounded in the retrieved review evidence
        """
    
        return prompt.strip()

    def generate_report(self, analysis_result: dict) -> str:
        prompt = self._build_prompt(analysis_result)
        report = self.llm.generate_text(prompt)

        predicted_class = str(analysis_result["predicted_class"]).lower()
        if predicted_class not in report.lower():
            return (
                f"Predicted price class: {analysis_result['predicted_class']}\n\n"
                f"The model predicts this class based on product text features and retrieved review evidence, "
                f"but the generated explanation was not fully aligned and was replaced by this safe summary."
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