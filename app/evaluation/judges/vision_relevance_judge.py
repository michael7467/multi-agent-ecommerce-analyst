from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.logging.logger import get_logger
from app.models.llm.llm_client import LLMClient
from app.evaluation.judges.llm_relevance_judge import _parse_yes_no

logger = get_logger("evaluation.vision_relevance")

_PROMPT_TEMPLATE = """
You are judging whether a product image is relevant to a search query, \
for evaluating an image-based product retrieval system.

Query: {query}

Look at the attached image. Is this image relevant to the query -- does \
it show a product that matches what the query is asking about, visually \
or in terms of product type/category?

Respond with exactly one word: "yes" or "no".
""".strip()


class VisionRelevanceJudge:
    """LLM-as-judge for image-text alignment, cached to disk exactly like
    LLMRelevanceJudge (see that file for the reasoning on why exceptions
    aren't cached but unparseable-but-real responses are -- same logic
    here, not repeated).

    Needs a vision-capable model -- gpt-4.1-mini is what the rest of this
    codebase defaults to for text, but wasn't confirmed here to support
    image input reliably, so this defaults to plain "gpt-4.1" unless told
    otherwise. Also unverified against a live account, same as
    generate_text_with_image itself -- test on one real image first.
    """

    def __init__(
        self,
        cache_path: str | Path = "data/eval/vision_relevance_cache.json",
        model: str = "gpt-4.1",
    ) -> None:
        self.llm = LLMClient(model=model)
        self.cache_path = Path(cache_path)
        self._cache: dict[str, dict] = self._load_cache()

    def _load_cache(self) -> dict[str, dict]:
        if not self.cache_path.exists():
            return {}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.error(f"Could not read vision relevance cache at {self.cache_path}, starting fresh", exc_info=True)
            return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=2)

    @staticmethod
    def _cache_key(query: str, image_url: str) -> str:
        # Keyed on the URL, not image bytes -- if the same URL could ever
        # point at different image content over time for this catalog,
        # this cache would go stale. Not a risk this project's static
        # product images are expected to have, but worth knowing if that
        # assumption changes.
        raw = f"{query}||{image_url}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def is_relevant(self, query: str, image_url: str) -> bool:
        if not image_url:
            return False

        key = self._cache_key(query, image_url)

        if key in self._cache:
            return self._cache[key]["relevant"]

        prompt = _PROMPT_TEMPLATE.format(query=query)

        try:
            raw = self.llm.generate_text_with_image(prompt, image_url)
        except Exception:
            # Not cached -- same reasoning as LLMRelevanceJudge: an API
            # failure is transient and should be retried next run, not
            # frozen into a permanent wrong answer. Vision calls are more
            # expensive and slower than text-only ones, which makes this
            # distinction matter even more here, not less.
            logger.error(f"Vision LLM call failed for query={query!r}, image_url={image_url!r}", exc_info=True)
            return False

        parsed = _parse_yes_no(raw)

        if parsed is None:
            logger.warning(f"Could not parse vision relevance judgment for query={query!r}, defaulting to not-relevant")
            parsed = False

        self._cache[key] = {
            "relevant": parsed,
            "query": query,
            "image_url": image_url,
        }
        self._save_cache()

        return parsed