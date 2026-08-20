from __future__ import annotations

from threading import Lock

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.logging.logger import get_logger
from app.observability.tracing import get_tracer
from app.config.paths import FEATURES_PATH


logger = get_logger("recommender.service")

_RECOMMENDER_STATE: tuple[pd.DataFrame, TfidfVectorizer, object] | None = None
_RECOMMENDER_LOCK = Lock()


def _load_recommender_state() -> tuple[pd.DataFrame, TfidfVectorizer, object]:
   
    global _RECOMMENDER_STATE

    if _RECOMMENDER_STATE is not None:
        return _RECOMMENDER_STATE

    with _RECOMMENDER_LOCK:
        if _RECOMMENDER_STATE is not None:
            return _RECOMMENDER_STATE

        df = pd.read_csv(FEATURES_PATH)

        df = df.reset_index(drop=True)

        for col in ["title", "categories", "description"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        df["combined_text"] = (
            df["title"] + " " +
            df["categories"] + " " +
            df["description"]
        )

        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        text_matrix = vectorizer.fit_transform(df["combined_text"])

        _RECOMMENDER_STATE = (df, vectorizer, text_matrix)

    return _RECOMMENDER_STATE


class RecommenderService:
    def __init__(self) -> None:
        self.df, self.vectorizer, self.text_matrix = _load_recommender_state()
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