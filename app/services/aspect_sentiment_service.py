from __future__ import annotations

import json
from typing import Literal
from transformers import pipeline

from app.models.llm.llm_client import LLMClient
from app.services.aspect_service import AspectService
from app.config.settings import settings
from app.logging.logger import get_logger
from app.observability.tracing import get_tracer
from app.prompts.system_prompts import ASPECT_SENTIMENT_SYSTEM_PROMPT
from app.prompts.agent_prompts import build_aspect_sentiment_user_prompt

logger = get_logger("aspect.sentiment")

AspectBackend = Literal["zero_shot", "llm"]

_ZSC_MODEL = None  # Singleton


class AspectSentimentService:
    def __init__(self, backend: AspectBackend = "zero_shot") -> None:
        self.backend = backend
        self.aspect_service = AspectService()
        self.tracer = get_tracer("app.aspect_sentiment")

        global _ZSC_MODEL

        if backend == "zero_shot":
            if _ZSC_MODEL is None:
                _ZSC_MODEL = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                )
            self.classifier = _ZSC_MODEL

        elif backend == "llm":
            self.llm = LLMClient(model=settings.llm_model)

        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def analyze_product_aspects(self, product_id: str, top_k: int = 3) -> dict:
        if not isinstance(product_id, str) or not product_id.strip():
            raise ValueError("product_id must be a non-empty string")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        with self.tracer.start_as_current_span("aspect_sentiment.analyze"):
            aspect_evidence = self.aspect_service.get_aspect_evidence(
                product_id=product_id,
                top_k=top_k,
            )

            results = {}
            for aspect, evidence in aspect_evidence.items():
                if self.backend == "zero_shot":
                    results[aspect] = self._analyze_aspect_zero_shot(aspect, evidence)
                else:
                    results[aspect] = self._analyze_aspect_llm(aspect, evidence)

            return results

    def _join_evidence(self, evidence: list[dict]) -> str:
        parts = []
        for ev in evidence:
            parts.append(
                f"Title: {ev.get('review_title', '')}\n"
                f"Text: {ev.get('review_text', '')}"
            )
        return "\n\n".join(parts).strip()

    def _normalize_label(self, label: str) -> str:
        label = label.lower()
        if label in {"positive", "negative"}:
            return label
        return "mixed"

    def _analyze_aspect_zero_shot(self, aspect: str, evidence: list[dict]) -> dict:
        if not evidence:
            return {"aspect": aspect, "label": "unknown", "score": 0.0, "method": "zero_shot"}

        text = self._join_evidence(evidence)
        candidate_labels = ["positive", "negative", "neutral"]

        try:
            result = self.classifier(text, candidate_labels=candidate_labels, multi_label=False)
            raw_label = result["labels"][0]
            score = float(result["scores"][0])
            final_label = self._normalize_label(raw_label)
        except Exception:
            logger.error("Zero-shot classifier failed", exc_info=True)
            return {"aspect": aspect, "label": "mixed", "score": 0.5, "method": "zero_shot_fallback"}

        return {
            "aspect": aspect,
            "label": final_label,
            "score": score,
            "method": "zero_shot",
        }

    def _analyze_aspect_llm(self, aspect: str, evidence: list[dict]) -> dict:
        if not evidence:
            return {"aspect": aspect, "label": "unknown", "score": 0.0, "method": "llm"}

        text = self._join_evidence(evidence)

        user_prompt = build_aspect_sentiment_user_prompt(aspect, text)

        try:
            raw = self.llm.generate_text(
                user_prompt,
                system_prompt=ASPECT_SENTIMENT_SYSTEM_PROMPT,
            )
            parsed = json.loads(raw)
            final_label = self._normalize_label(parsed.get("label", "mixed"))
            score = float(parsed.get("score", 0.5))
        except Exception:
            # Covers a malformed/missing JSON response AND a present but
            # non-numeric "score" (e.g. the LLM returns "score": "high") --
            # the latter used to sit outside this try/except, so a
            # well-formed-but-wrong-shaped response would raise uncaught
            # instead of falling back like an actually-malformed one did.
            logger.error("LLM returned an unusable aspect sentiment response", exc_info=True)
            return {"aspect": aspect, "label": "mixed", "score": 0.5, "method": "llm_fallback"}

        return {
            "aspect": aspect,
            "label": final_label,
            "score": score,
            "method": "llm",
        }