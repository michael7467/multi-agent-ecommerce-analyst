from __future__ import annotations

from app.services.cache_service import CacheService
from app.rag.qdrant_retriever import QdrantRetriever
from app.logging.logger import get_logger
from app.observability.tracing import get_tracer

logger = get_logger("rag.service")


class RAGService:
    def __init__(self) -> None:
        self.retriever = QdrantRetriever()
        self.cache = CacheService()
        self.tracer = get_tracer("app.rag_service")

    def get_product_evidence(
        self,
        product_id: str,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:

        if not isinstance(product_id, str) or not product_id.strip():
            raise ValueError("product_id must be a non-empty string")

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        cache_payload = {
            "product_id": product_id,
            "query": query,
            "top_k": top_k,
        }

        with self.tracer.start_as_current_span("rag.get_product_evidence") as span:
            span.set_attribute("product_id", product_id)
            span.set_attribute("query", query)
            span.set_attribute("top_k", top_k)

            # --- Cache lookup ---
            cached = self.cache.get_json("rag:evidence", cache_payload)
            if cached is not None:
                logger.debug(f"Cache hit for product_id={product_id}, query='{query}'")
                span.set_attribute("cache_hit", True)
                return cached

            logger.debug(f"Cache miss for product_id={product_id}, query='{query}'")
            span.set_attribute("cache_hit", False)

            # --- Retrieval ---
            try:
                results = self.retriever.search(
                    query=query,
                    top_k=top_k,
                    product_id=product_id,
                )
            except Exception:
                logger.error(
                    f"Qdrant retrieval failed for product_id={product_id}, query='{query}'",
                    exc_info=True,
                )
                span.set_attribute("retrieval_error", True)
                return []

            span.set_attribute("retrieval_error", False)
            span.set_attribute("retrieved_count", len(results))

            # --- Normalize evidence ---
            evidence = []
            for item in results:
                evidence.append(
                    {
                        "product_id": item.get("product_id", ""),
                        "title": item.get("title", ""),
                        "review_text": item.get("review_text", ""),
                        "review_title": item.get("review_title", ""),
                        "categories": item.get("categories", ""),
                        "score": float(item.get("score", 0.0)),
                    }
                )

            span.set_attribute("evidence_count", len(evidence))

            # --- Cache write ---
            try:
                self.cache.set_json(
                    "rag:evidence",
                    cache_payload,
                    evidence,
                    ttl_seconds=1800,
                )
                logger.debug(
                    f"Cached evidence for product_id={product_id}, query='{query}' "
                    f"(count={len(evidence)})"
                )
            except Exception:
                logger.error(
                    f"Failed to write evidence to cache for product_id={product_id}",
                    exc_info=True,
                )

            return evidence
