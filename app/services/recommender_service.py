from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


FEATURES_PATH = "data/processed/electronics_labeled.csv"


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

    def recommend_similar_products(self, product_id: str, top_k: int = 5) -> list[dict]:
        matches = self.df[self.df["product_id"].astype(str) == str(product_id)]
        if matches.empty:
            raise ValueError(f"Product not found: {product_id}")

        product_idx = matches.index[0]

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
            recommendations.append(
                {
                    "product_id": row["product_id"],
                    "title": row.get("title", ""),
                    "categories": row.get("categories", ""),
                    "price": row.get("price", None),
                    "predicted_class": row.get("price_class", ""),
                    "similarity_score": float(similarity_scores[idx]),
                }
            )

            if len(recommendations) >= top_k:
                break

        return recommendations


if __name__ == "__main__":
    service = RecommenderService()

    results = service.recommend_similar_products(
        product_id="B09SPZPDJK",
        top_k=5,
    )

    print("\n=== SIMILAR PRODUCTS ===\n")
    for item in results:
        print(item)