from __future__ import annotations

import pandas as pd
from app.models.forecasting.predict import PricePredictor
from app.services.rag_service import RAGService
from app.logging.logger import get_logger
from app.observability.tracing import get_tracer

FEATURES_PATH = "data/processed/electronics_labeled.csv"
logger = get_logger("analysis.service")


class AnalysisService:
    def __init__(self) -> None:
        self.predictor = PricePredictor()
        self.rag_service = RAGService()
        self.tracer = get_tracer("app.analysis_service")

        self.features_df = pd.read_csv(FEATURES_PATH)

        required = [
            "product_id", "review_count", "avg_rating", "rating_std",
            "verified_purchase_ratio", "avg_review_length", "review_time_span",
            "title", "categories", "price"
        ]
        for col in required:
            if col not in self.features_df.columns:
                raise RuntimeError(f"Missing required feature column: {col}")

        # Faster lookups
        self.features_df["product_id"] = self.features_df["product_id"].astype(str)
        self.features_df.set_index("product_id", inplace=True)

    def get_product_row(self, product_id: str) -> dict:
        if not isinstance(product_id, str) or not product_id.strip():
            raise ValueError("product_id must be a non-empty string")

        try:
            return self.features_df.loc[str(product_id)].to_dict()
        except KeyError:
            raise ValueError(f"Product not found: {product_id}")

    def analyze_product(self, product_id: str, query: str, top_k: int = 3) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        with self.tracer.start_as_current_span("analysis_service.analyze_product") as span:
            span.set_attribute("product_id", product_id)
            span.set_attribute("query", query)
            span.set_attribute("top_k", top_k)

            row = self.get_product_row(product_id)

            model_input = {
                "review_count": row["review_count"],
                "avg_rating": row["avg_rating"],
                "rating_std": row["rating_std"],
                "verified_purchase_ratio": row["verified_purchase_ratio"],
                "avg_review_length": row["avg_review_length"],
                "review_time_span": row["review_time_span"],
                "title": row.get("title", ""),
                "categories": row.get("categories", ""),
            }

            try:
                prediction = self.predictor.predict(model_input)
            except Exception:
                logger.error("Price prediction failed", exc_info=True)
                prediction = {"predicted_class": None}

            try:
                evidence = self.rag_service.get_product_evidence(
                    product_id=product_id,
                    query=query,
                    top_k=top_k,
                )
            except Exception:
                logger.error("RAG evidence retrieval failed", exc_info=True)
                evidence = []

            result = {
                "product_id": product_id,
                "title": row.get("title", ""),
                "categories": row.get("categories", ""),
                "price": row.get("price", None),
                "predicted_class": prediction["predicted_class"],
                "evidence": evidence,
            }

            span.set_attribute("has_evidence", len(evidence) > 0)
            return result
