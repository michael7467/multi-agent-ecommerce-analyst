from __future__ import annotations

import math
import re


def is_relevant(text: str, keywords: list[str]) -> bool:

    text_lower = text.lower()
    return any(
        re.search(r"\b" + re.escape(kw.lower()) + r"\b", text_lower)
        for kw in keywords
    )


class KeywordRelevanceJudge:

    def is_relevant(self, query: str, review_text: str, relevant_keywords: list[str] | None = None) -> bool:
        return is_relevant(review_text, relevant_keywords or [])


def recall_at_k(retrieved_relevance: list[bool], total_relevant: int) -> float | None:
 
    if total_relevant == 0:
        return None
    found = sum(retrieved_relevance)
    return min(found / total_relevant, 1.0)


def reciprocal_rank(retrieved_relevance: list[bool]) -> float:
    """1/rank of the first relevant item; 0.0 if none found in the list."""
    for i, relevant in enumerate(retrieved_relevance, start=1):
        if relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_relevance: list[bool], total_relevant: int | None = None) -> float:
 
    def dcg(relevances: list[float]) -> float:
        return sum(
            rel / math.log2(i + 1)
            for i, rel in enumerate(relevances, start=1)
        )

    k = len(retrieved_relevance)
    actual = dcg([1.0 if r else 0.0 for r in retrieved_relevance])

    if total_relevant is None:
        ideal_relevance = sorted((1.0 if r else 0.0 for r in retrieved_relevance), reverse=True)
    else:
        num_ideal = min(k, total_relevant)
        ideal_relevance = [1.0] * num_ideal + [0.0] * (k - num_ideal)

    ideal = dcg(ideal_relevance)
    return actual / ideal if ideal > 0 else 0.0


def average_precision(retrieved_relevance: list[bool]) -> float:
  
    hits = 0
    precisions = []
    for i, relevant in enumerate(retrieved_relevance, start=1):
        if relevant:
            hits += 1
            precisions.append(hits / i)
    return sum(precisions) / hits if hits else 0.0


def keyword_coverage(retrieved_texts: list[str], relevant_keywords: list[str]) -> float:
 
    if not relevant_keywords:
        return 1.0
    combined = " ".join(retrieved_texts).lower()
    covered = sum(
        1 for kw in relevant_keywords
        if re.search(r"\b" + re.escape(kw.lower()) + r"\b", combined)
    )
    return covered / len(relevant_keywords)