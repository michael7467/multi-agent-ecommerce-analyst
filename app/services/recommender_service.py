from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.logging.logger import get_logger
from app.observability.tracing import get_tracer
from app.config.paths import FEATURES_PATH


logger = get_logger("recommender.service")


class RecommenderService:
    def __init__(self) -> None:
        self.df = pd.read_csv(FEATURES_PATH).copy()

        for col in ["title", "categories", "description"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("").astype(str)

        self.df["combined_text"] = (
            self.df["title"] + " " +
            self.df["categories"] + " " +
            self.df["description"]
        )

        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.text_matrix = self.vectorizer.fit_transform(self.df["combined_text"])

        self.tracer = get_tracer("app.recommender_service")

    def recommend_similar_products(self, product_id: str, top_k: int = 5) -> list[dict]:
        if not isinstance(product_id, str) or not product_id.strip():
            raise ValueError("product_id must be a non-empty string")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        with self.tracer.start_as_current_span("recommender.recommend") as span:
            span.set_attribute("product_id", product_id)
            span.set_attribute("top_k", top_k)

            matches = self.df[self.df["product_id"].astype(str) == str(product_id)]
            if matches.empty:
                logger.error(f"Product not found: {product_id}")
                span.set_attribute("product_found", False)
                raise ValueError(f"Product not found: {product_id}")

            span.set_attribute("product_found", True)

            product_idx = matches.index[0]

            # Compute similarity
            similarity_scores = cosine_similarity(
                self.text_matrix[product_idx],
                self.text_matrix
            ).flatten()

            similar_indices = similarity_scores.argsort()[::-1]

            recommendations = []
            for idx in similar_indices:
                if idx == product_idx:
                    continue

                row = self.df.iloc[idx]
                score = float(similarity_scores[idx])

                recommendations.append(
                    {
                        "product_id": row["product_id"],
                        "title": row.get("title", ""),
                        "categories": row.get("categories", ""),
                        "price": row.get("price", None),
                        "predicted_class": row.get("price_class", ""),
                        "similarity_score": score,
                    }
                )

                if len(recommendations) >= top_k:
                    break

            span.set_attribute("recommendation_count", len(recommendations))
            logger.debug(
                f"Generated {len(recommendations)} recommendations for product_id={product_id}"
            )

            return recommendations
