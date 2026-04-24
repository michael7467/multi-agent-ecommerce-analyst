from __future__ import annotations

import pandas as pd
from app.config.paths import FEATURES_PATH, REVIEWS_PATH



class TrendDetectionService:
    def __init__(self) -> None:
        self.features_df = pd.read_csv(FEATURES_PATH)
        self.reviews_df = pd.read_csv(REVIEWS_PATH)

    def analyze(self) -> dict:
        df = self.reviews_df.copy()

        if "review_timestamp" not in df.columns:
            raise ValueError("Trend detection requires review_timestamp in reviews data.")

        df["review_timestamp"] = pd.to_numeric(df["review_timestamp"], errors="coerce")
        df = df.dropna(subset=["review_timestamp"]).copy()

        df["review_datetime"] = pd.to_datetime(df["review_timestamp"], unit="ms", errors="coerce")

        missing_dt = df["review_datetime"].isna()
        if missing_dt.any():
            df.loc[missing_dt, "review_datetime"] = pd.to_datetime(
                df.loc[missing_dt, "review_timestamp"],
                unit="s",
                errors="coerce",
            )

        df = df.dropna(subset=["review_datetime"]).copy()
        df["year_month"] = df["review_datetime"].dt.to_period("M").astype(str)

        merged = df.merge(
            self.features_df[["product_id", "categories"]],
            on="product_id",
            how="left",
        )

        merged["categories"] = merged["categories"].fillna("Unknown")
        merged["main_category"] = merged["categories"].astype(str).str.split("|").str[0].str.strip()

        rising_categories = self._detect_rising_categories(merged)
        declining_categories = self._detect_declining_categories(merged)
        seasonal_patterns = self._detect_seasonal_patterns(merged)
        emerging_complaints = self._detect_emerging_complaints(merged)

        return {
            "rising_categories": rising_categories,
            "declining_categories": declining_categories,
            "seasonal_patterns": seasonal_patterns,
            "emerging_complaints": emerging_complaints,
        }

    def _monthly_category_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.groupby(["main_category", "year_month"])
            .size()
            .reset_index(name="review_count")
            .sort_values(["main_category", "year_month"])
        )

    def _trend_score(self, counts: list[int]) -> float:
        if len(counts) < 2:
            return 0.0
        first = counts[0]
        last = counts[-1]
        if first == 0:
            return float(last)
        return (last - first) / first

    def _detect_rising_categories(self, df: pd.DataFrame, top_k: int = 5) -> list[dict]:
        monthly = self._monthly_category_counts(df)

        rows = []
        for category, group in monthly.groupby("main_category"):
            counts = group["review_count"].tolist()
            score = self._trend_score(counts)
            rows.append(
                {
                    "category": category,
                    "trend_score": score,
                    "latest_review_count": int(group["review_count"].iloc[-1]),
                }
            )

        result = pd.DataFrame(rows).sort_values("trend_score", ascending=False)
        return result.head(top_k).to_dict(orient="records")

    def _detect_declining_categories(self, df: pd.DataFrame, top_k: int = 5) -> list[dict]:
        monthly = self._monthly_category_counts(df)

        rows = []
        for category, group in monthly.groupby("main_category"):
            counts = group["review_count"].tolist()
            score = self._trend_score(counts)
            rows.append(
                {
                    "category": category,
                    "trend_score": score,
                    "latest_review_count": int(group["review_count"].iloc[-1]),
                }
            )

        result = pd.DataFrame(rows).sort_values("trend_score", ascending=True)
        return result.head(top_k).to_dict(orient="records")

    def _detect_seasonal_patterns(self, df: pd.DataFrame, top_k: int = 5) -> list[dict]:
        seasonal = df.copy()
        seasonal["month"] = seasonal["review_datetime"].dt.month

        grouped = (
            seasonal.groupby(["main_category", "month"])
            .size()
            .reset_index(name="review_count")
        )

        rows = []
        for category, group in grouped.groupby("main_category"):
            peak_row = group.sort_values("review_count", ascending=False).iloc[0]
            rows.append(
                {
                    "category": category,
                    "peak_month": int(peak_row["month"]),
                    "peak_review_count": int(peak_row["review_count"]),
                }
            )

        result = pd.DataFrame(rows).sort_values("peak_review_count", ascending=False)
        return result.head(top_k).to_dict(orient="records")

    def _detect_emerging_complaints(self, df: pd.DataFrame, top_k: int = 5) -> list[dict]:
        complaint_keywords = [
            "broken",
            "broke",
            "bad",
            "poor",
            "problem",
            "issue",
            "issues",
            "defect",
            "defective",
            "refund",
            "return",
            "hollow",
            "disconnect",
            "slow",
            "failed",
            "does not work",
        ]

        temp = df.copy()
        temp["review_text"] = temp["review_text"].fillna("").astype(str).str.lower()

        rows = []
        for keyword in complaint_keywords:
            matched = temp[temp["review_text"].str.contains(keyword, na=False)]
            if matched.empty:
                continue

            monthly = (
                matched.groupby("year_month")
                .size()
                .reset_index(name="count")
                .sort_values("year_month")
            )

            score = self._trend_score(monthly["count"].tolist())
            rows.append(
                {
                    "complaint": keyword,
                    "trend_score": score,
                    "latest_count": int(monthly["count"].iloc[-1]),
                }
            )

        if not rows:
            return []

        result = pd.DataFrame(rows).sort_values("trend_score", ascending=False)
        return result.head(top_k).to_dict(orient="records")