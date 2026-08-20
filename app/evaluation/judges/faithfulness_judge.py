from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from app.logging.logger import get_logger
from app.models.llm.llm_client import LLMClient

logger = get_logger("evaluation.faithfulness")

_PROMPT_TEMPLATE = """
You are fact-checking a generated product analysis report against the \
data it was supposed to be based on.

Available data (the ONLY source of truth -- treat anything not stated \
here as unverifiable, even if it sounds plausible):
{context}

Report to check:
{report}

Break the report into its distinct factual claims. Skip purely \
stylistic, transitional, or framing sentences ("Overall, this is a \
solid choice") -- only include sentences that assert something as fact \
about the product, its reviews, its price, or its performance.

For each claim, judge whether it is directly supported by the available \
data above.

Return a JSON array only, no markdown formatting, no commentary before \
or after it:
[{{"claim": "...", "grounded": true}}, {{"claim": "...", "grounded": false}}]
""".strip()


def _parse_claims(text: str) -> list[dict] | None:
    """Tolerates markdown code fences and leading/trailing commentary the
    model might add despite instructions not to -- same defensive
    tolerance as every other LLM-JSON parse in this project
    (planning_agent's plan extraction, critic_agent's score parsing).
    Returns None if nothing usable could be recovered, not an empty list
    -- "couldn't parse this" and "the model said there are zero claims"
    need to stay distinguishable to the caller.
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", text.strip())

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    if not isinstance(parsed, list):
        return None

    claims = []
    for item in parsed:
        if isinstance(item, dict) and "claim" in item and "grounded" in item:
            claims.append({"claim": str(item["claim"]), "grounded": bool(item["grounded"])})

    return claims


class FaithfulnessJudge:
    """Claim-level faithfulness checking: extracts factual claims from a
    report and judges each one against the context it was generated from,
    in one LLM call rather than one call per claim.

    Worth knowing before using this: CriticAgent already scores
    "Hallucination Risk" 1-10 as part of its existing critique in
    production. This isn't a duplicate of that -- it operates at a
    different level of granularity (which specific claims are
    unsupported, not a single holistic number) -- but you now have two
    different tools measuring a similar thing, and that's worth being
    aware of rather than discovering by surprise.
    """

    def __init__(
        self,
        cache_path: str | Path = "data/eval/faithfulness_cache.json",
        model: str = "gpt-4.1-mini",
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
            logger.error(f"Could not read faithfulness cache at {self.cache_path}, starting fresh", exc_info=True)
            return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=2)

    @staticmethod
    def _cache_key(context: str, report: str) -> str:
        raw = f"{context}||{report}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def check(self, context: str, report: str) -> dict:
        """Returns {"claims": [...], "faithfulness": float | None,
        "hallucination_rate": float | None, "n_claims": int}.

        faithfulness/hallucination_rate are None (not 0.0/1.0) when there
        are zero extractable claims -- e.g. a very short report, or the
        guardrail's own canned fallback message. Nothing to judge isn't
        the same as everything being grounded or everything being wrong.
        """
        key = self._cache_key(context, report)

        if key in self._cache:
            return self._cache[key]

        prompt = _PROMPT_TEMPLATE.format(context=context, report=report)

        try:
            raw = self.llm.generate_text(prompt)
        except Exception:
 
            logger.error("Faithfulness LLM call failed", exc_info=True)
            return {"claims": [], "faithfulness": None, "hallucination_rate": None, "n_claims": 0, "error": "llm_call_failed"}

        claims = _parse_claims(raw)

        if claims is None:
            logger.warning("Could not parse claims from faithfulness response")
            return {"claims": [], "faithfulness": None, "hallucination_rate": None, "n_claims": 0, "error": "unparseable_response"}

        n = len(claims)
        if n == 0:
            result = {"claims": [], "faithfulness": None, "hallucination_rate": None, "n_claims": 0}
        else:
            grounded = sum(1 for c in claims if c["grounded"])
            result = {
                "claims": claims,
                "faithfulness": grounded / n,
                "hallucination_rate": (n - grounded) / n,
                "n_claims": n,
            }

        self._cache[key] = result
        self._save_cache()
        return result
