from __future__ import annotations

from typing import Literal

from transformers import pipeline

from app.models.llm.llm_client import LLMClient
from app.services.aspect_service import AspectService
from app.core.config import settings

AspectBackend = Literal["zero_shot", "llm"]


class AspectSentimentService:
    def __init__(self, backend: AspectBackend = "zero_shot") -> None:
        self.backend = backend
        self.aspect_service = AspectService()

        if backend == "zero_shot":
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
            )
        elif backend == "llm":
            self.llm = LLMClient(model=settings.llm_model)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def analyze_product_aspects(self, product_id: str, top_k: int = 3) -> dict:
        aspect_evidence = self.aspect_service.get_aspect_evidence(
            product_id=product_id,
            top_k=top_k,
        )

        results = {}
        for aspect, evidence in aspect_evidence.items():
            if self.backend == "zero_shot":
                results[aspect] = self._analyze_aspect_zero_shot(aspect, evidence)
            elif self.backend == "llm":
                results[aspect] = self._analyze_aspect_llm(aspect, evidence)

        return results

    def _join_evidence(self, evidence: list[dict]) -> str:
        texts = []
        for ev in evidence:
            review_title = ev.get("review_title", "")
            review_text = ev.get("review_text", "")
            texts.append(f"Title: {review_title}\nText: {review_text}")
        return "\n\n".join(texts).strip()

    def _label_from_score(self, score: float) -> str:
        if score >= 0.60:
            return "positive"
        if score <= 0.40:
            return "negative"
        return "mixed"

    def _analyze_aspect_zero_shot(self, aspect: str, evidence: list[dict]) -> dict:
        if not evidence:
            return {
                "aspect": aspect,
                "label": "unknown",
                "score": 0.0,
                "method": "zero_shot",
            }

        text = self._join_evidence(evidence)
        candidate_labels = ["positive", "negative", "neutral"]

        result = self.classifier(
            text,
            candidate_labels=candidate_labels,
            multi_label=False,
        )

        top_label = result["labels"][0]
        top_score = float(result["scores"][0])

        final_label = "mixed" if top_label == "neutral" else top_label

        return {
            "aspect": aspect,
            "label": final_label,
            "score": top_score,
            "method": "zero_shot",
        }

    def _analyze_aspect_llm(self, aspect: str, evidence: list[dict]) -> dict:
        if not evidence:
            return {
                "aspect": aspect,
                "label": "unknown",
                "score": 0.0,
                "method": "llm",
            }

        text = self._join_evidence(evidence)

        prompt = f"""
You are an aspect-based sentiment analyst.

Your task is to judge sentiment for one product aspect using only the provided review evidence.

Aspect: {aspect.replace("_", " ")}

Rules:
- Use only the review evidence.
- Output one of: positive, negative, mixed
- Also output a confidence score from 0.0 to 1.0
- Output valid JSON only
- Do not include markdown

Review evidence:
{text}

Return:
{{
  "label": "positive",
  "score": 0.85
}}
""".strip()

        raw = self.llm.generate_text(prompt)

        import json
        parsed = json.loads(raw)

        return {
            "aspect": aspect,
            "label": parsed.get("label", "mixed"),
            "score": float(parsed.get("score", 0.5)),
            "method": "llm",
        }