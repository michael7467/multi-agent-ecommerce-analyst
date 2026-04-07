from __future__ import annotations

import pandas as pd

from app.models.forecasting.predict import PricePredictor
from app.services.rag_service import RAGService


FEATURES_PATH = "data/processed/electronics_labeled.csv"


class AnalysisService:
    def __init__(self) -> None:
        self.predictor = PricePredictor()
        self.rag_service = RAGService()
        self.features_df = pd.read_csv(FEATURES_PATH)

    def get_product_row(self, product_id: str) -> dict:
        matches = self.features_df[self.features_df["product_id"].astype(str) == str(product_id)]
        if matches.empty:
            raise ValueError(f"Product not found: {product_id}")

        row = matches.iloc[0].to_dict()
        return row

    def analyze_product(self, product_id: str, query: str, top_k: int = 3) -> dict:
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

        prediction = self.predictor.predict(model_input)
        evidence = self.rag_service.get_product_evidence(
            product_id=product_id,
            query=query,
            top_k=top_k,
        )

        return {
            "product_id": product_id,
            "title": row.get("title", ""),
            "categories": row.get("categories", ""),
            "price": row.get("price", None),
            "predicted_class": prediction["predicted_class"],
            "evidence": evidence,
        }


if __name__ == "__main__":
    service = AnalysisService()

    sample_product_id = "B09SPZPDJK"
    query = "sound quality and noise cancellation"

    result = service.analyze_product(
        product_id=sample_product_id,
        query=query,
        top_k=3,
    )

    print("\nAnalysis Result:")
    print(result)