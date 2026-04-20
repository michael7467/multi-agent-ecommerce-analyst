from __future__ import annotations

from app.services.rag_service import RAGService
from app.logging.logger import get_logger
from app.observability.tracing import get_tracer

logger = get_logger("aspect.service")


class AspectService:
    ASPECT_QUERIES = {
        "sound_quality": "sound quality audio bass treble volume hollow sound",
        "battery_life": "battery life charging power lasts long battery",
        "comfort": "comfort comfortable fit bulky wear earpads head",
        "build_quality": "build quality design material premium solid construction",
        "durability": "durability durable broke broken long lasting quality issue",
        "price_value": "price value worth money expensive cheap overpriced deal",
    }

    def __init__(self) -> None:
        self.rag_service = RAGService()
        self.tracer = get_tracer("app.aspect_service")

    def get_aspect_evidence(self, product_id: str, top_k: int = 3) -> dict[str, list[dict]]:
        if not isinstance(product_id, str) or not product_id.strip():
            raise ValueError("product_id must be a non-empty string")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        aspect_results: dict[str, list[dict]] = {}

        with self.tracer.start_as_current_span("aspect_service.get_aspect_evidence") as span:
            span.set_attribute("product_id", product_id)
            span.set_attribute("top_k", top_k)

            for aspect, query in self.ASPECT_QUERIES.items():
                try:
                    evidence = self.rag_service.get_product_evidence(
                        product_id=product_id,
                        query=query,
                        top_k=top_k,
                    )
                except Exception:
                    logger.error(f"RAG retrieval failed for aspect '{aspect}'", exc_info=True)
                    evidence = []

                aspect_results[aspect] = evidence
                span.set_attribute(f"{aspect}_count", len(evidence))

        return aspect_results
