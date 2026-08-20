from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from app.logging.logger import get_logger
from app.models.llm.llm_client import LLMClient
from app.observability.metrics import parse_failure_total, fallback_usage_total

logger = get_logger("evaluation.llm_relevance")

_PROMPT_TEMPLATE = """
You are judging whether a customer review is relevant to a search query, \
for evaluating a product review retrieval system.

Query: {query}

Review: {review_text}

A review is relevant if it discusses the topic, aspect, or question the \
query is asking about -- even if it uses different words than the query \
(for example, a query about "battery life" is relevant to a review about \
"lasts all day on one charge", even though neither review uses the word \
"life"). A review that merely mentions a related word without actually \
addressing the query's topic is not relevant.

Respond with exactly one word: "yes" or "no".
""".strip()


def _parse_yes_no(text: str) -> bool | None:
    """None means genuinely unparseable -- distinct from a confident "no",
    since callers should be able to tell "the LLM said not relevant" apart
    from "the LLM's response didn't parse", the same distinction made
    everywhere else in this project between a real result and a fallback.
    """
    cleaned = text.strip().lower()
    if re.match(r"^\W*yes\b", cleaned):
        return True
    if re.match(r"^\W*no\b", cleaned):
        return False
    if "yes" in cleaned and "no" not in cleaned:
        return True
    if "no" in cleaned and "yes" not in cleaned:
        return False
    return None


class LLMRelevanceJudge:
    """LLM-as-judge relevance checking, cached to disk.

    Interface matches KeywordRelevanceJudge (see retrieval_metrics.py) so
    RetrievalEvaluator can use either one interchangeably: both expose
    is_relevant(query, review_text, relevant_keywords=None). This one
    ignores relevant_keywords entirely -- the whole point is not needing
    them -- but accepts the parameter so the two are drop-in compatible.
    """

    def __init__(
        self,
        cache_path: str | Path = "data/eval/relevance_cache.json",
        model: str = "gpt-4.1-mini",
    ) -> None:
        self.llm = LLMClient(model=model)
        self.cache_path = Path(cache_path)
        self._cache: dict[str, dict] = self._load_cache()
        self._new_judgments_this_run = 0

    def _load_cache(self) -> dict[str, dict]:
        if not self.cache_path.exists():
            return {}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.error(f"Could not read relevance cache at {self.cache_path}, starting fresh", exc_info=True)
            return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=2)

    @staticmethod
    def _cache_key(query: str, review_text: str) -> str:
        raw = f"{query}||{review_text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def is_relevant(self, query: str, review_text: str, relevant_keywords: list[str] | None = None) -> bool:
        key = self._cache_key(query, review_text)

        if key in self._cache:
            return self._cache[key]["relevant"]

        prompt = _PROMPT_TEMPLATE.format(query=query, review_text=review_text)

        try:
            raw = self.llm.generate_text(prompt)
        except Exception:
  
            logger.error(f"LLM call failed for query={query!r}", exc_info=True)
            fallback_usage_total.labels(component="llm_relevance_judge", reason="llm_call_failed").inc()
            return False

        parsed = _parse_yes_no(raw)

        if parsed is None:
       
            logger.warning(f"Could not parse relevance judgment for query={query!r}, defaulting to not-relevant")
            parse_failure_total.labels(component="llm_relevance_judge").inc()
            fallback_usage_total.labels(component="llm_relevance_judge", reason="unparseable_response").inc()
            parsed = False

        self._cache[key] = {
            "relevant": parsed,
            "query": query,
            "review_snippet": review_text[:200],
        }
        self._new_judgments_this_run += 1

        self._save_cache()

        return parsed