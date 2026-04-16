from __future__ import annotations

from app.models.llm.llm_client import LLMClient


class ReportService:
    def __init__(self) -> None:
        self.llm = LLMClient(model="gpt-4.1-mini")

    def _build_prompt(self, analysis_result: dict) -> str:
        product_id = analysis_result.get("product_id", "")
        title = analysis_result.get("title", "")
        categories = analysis_result.get("categories", "")
        price = analysis_result.get("price", "")
        predicted_class = analysis_result.get("predicted_class", None)
        evidence = analysis_result.get("evidence", [])
        recommendations = analysis_result.get("recommendations", [])
        image_similar_products = analysis_result.get("image_similar_products", [])
        aspect_summaries = analysis_result.get("aspect_summaries", {})

        sentiment = analysis_result.get("sentiment", {})
        avg_sentiment_score = sentiment.get("avg_sentiment_score", 0.0)
        positive_review_ratio = sentiment.get("positive_review_ratio", 0.0)
        neutral_review_ratio = sentiment.get("neutral_review_ratio", 0.0)
        negative_review_ratio = sentiment.get("negative_review_ratio", 0.0)

        evidence_text = []
        for i, ev in enumerate(evidence, start=1):
            evidence_text.append(
                f"""Evidence {i}:
Review title: {ev.get("review_title", "")}
Review text: {ev.get("review_text", "")}
Similarity score: {ev.get("score", 0):.4f}
"""
            )

        recommendation_text = []
        for i, rec in enumerate(recommendations, start=1):
            recommendation_text.append(
                f"""Recommendation {i}:
Product ID: {rec.get("product_id", "")}
Title: {rec.get("title", "")}
Similarity score: {rec.get("similarity_score", 0):.4f}
Predicted class: {rec.get("predicted_class", "")}
"""
            )

        image_text = []
        for i, item in enumerate(image_similar_products, start=1):
            image_text.append(
                f"""Visual Match {i}:
Product ID: {item.get("product_id", "")}
Title: {item.get("title", "")}
Similarity score: {item.get("similarity_score", 0):.4f}
"""
            )

        aspect_text = []
        for aspect, payload in aspect_summaries.items():
            aspect_text.append(f"{aspect}: {payload.get('summary', '')}")

        prediction_block = ""
        if predicted_class is not None:
            prediction_block = f"Predicted price class: {predicted_class}"

        prompt = f"""
You are an expert e-commerce AI analyst.

Your task is to answer the user's query using the available analysis results.

Rules:
- Use only the provided evidence and analysis results.
- Do not invent facts.
- If a price-class prediction is provided, treat it as fixed and do not change it.
- If a field is missing, do not pretend it exists.
- Output plain text only.

Product Information:
- Product ID: {product_id}
- Title: {title}
- Categories: {categories}
- Price: {price}
{prediction_block}

Sentiment Summary:
- Average sentiment score: {avg_sentiment_score:.3f}
- Positive reviews: {positive_review_ratio:.2%}
- Neutral reviews: {neutral_review_ratio:.2%}
- Negative reviews: {negative_review_ratio:.2%}

Aspect Summaries:
{"; ".join(aspect_text)}

Retrieved Evidence:
{"".join(evidence_text)}

Text Recommendations:
{"".join(recommendation_text)}

Image-Based Similar Products:
{"".join(image_text)}

Write a professional answer to the user's request.
If a predicted class exists, include it clearly.
If recommendations exist, mention them when relevant.
If visual matches exist, mention them when relevant.
If sentiment exists, use it to describe customer perception.
""".strip()

        return prompt

    def generate_report(self, analysis_result: dict) -> str:
        prompt = self._build_prompt(analysis_result)
        report = self.llm.generate_text(prompt)

        predicted_class = analysis_result.get("predicted_class", None)
        if predicted_class is not None:
            if str(predicted_class).lower() not in report.lower():
                return (
                    f"Predicted price class: {predicted_class}\n\n"
                    f"The model predicts this class based on product features, sentiment signals, "
                    f"and retrieved evidence. However, the generated explanation was not fully aligned, "
                    f"so this safe summary is returned instead."
                )

        return report


if __name__ == "__main__":
    from app.agents.dynamic_orchestrator import DynamicOrchestrator

    orchestrator = DynamicOrchestrator()

    result = orchestrator.run(
        product_id="B09SPZPDJK",
        query="What do customers think about sound quality?",
        top_k=3,
    )

    report_service = ReportService()
    report = report_service.generate_report(result["final_output"])

    print("\n=== FINAL LLM REPORT ===\n")
    print(report)