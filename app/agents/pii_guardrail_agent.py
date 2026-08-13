from __future__ import annotations

import re

from app.agents.base_agent import BaseAgent
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent

logger = get_logger("agents.pii_guardrail")


# Deliberately scoped to email + phone specifically -- well-defined,
# regex-detectable formats, matching the existing guardrail_agent's own
# pattern-matching style rather than introducing a new detection paradigm.
# Physical addresses are real PII too, but far harder to detect reliably
# without heavy false-positive risk against this domain's own product
# text (model numbers, dimensions, ratings) -- out of scope here.
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\w)"
)


def _redact(text: str) -> tuple[str, list[str]]:
    """
    Returns (redacted_text, list of PII types found -- e.g. ["email"]).
    Word-boundary-anchored patterns, verified directly against realistic
    product-review text (model numbers, ratings, durations) to confirm
    they don't false-positive on this domain's own alphanumeric codes.
    """
    found: list[str] = []

    if _EMAIL_PATTERN.search(text):
        found.append("email")
        text = _EMAIL_PATTERN.sub("[EMAIL REDACTED]", text)

    if _PHONE_PATTERN.search(text):
        found.append("phone")
        text = _PHONE_PATTERN.sub("[PHONE REDACTED]", text)

    return text, found


class PIIGuardrailAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="PIIGuardrailAgent")

    @traced_agent("PIIGuardrailAgent.run")
    def run(self, report: str | None, evidence: list[dict] | None) -> dict:
        """
        Scans and redacts both report text and evidence review snippets --
        both are text that actually reaches the end user in final_output,
        regardless of which stage of the pipeline any PII originally
        entered through.

        Returns {} (no state changes) when there's nothing to scan or
        nothing found -- same "silent no-op when nothing to do" pattern
        already used by guardrail_agent/critic_agent, not a special case
        invented for this agent.
        """
        result: dict = {}
        pii_types_found: set[str] = set()

        if report:
            redacted_report, found = _redact(report)
            if found:
                pii_types_found.update(found)
                result["report"] = redacted_report
                logger.warning(
                    "PII redacted from report",
                    extra={"pii_types": found},
                )

        if evidence:
            redacted_evidence = []
            any_evidence_redacted = False
            for item in evidence:
                item = dict(item)
                review_text = item.get("review_text", "")
                if review_text:
                    redacted_text, found = _redact(review_text)
                    if found:
                        pii_types_found.update(found)
                        any_evidence_redacted = True
                        item["review_text"] = redacted_text
                        logger.warning(
                            "PII redacted from evidence review_text",
                            extra={"pii_types": found, "product_id": item.get("product_id")},
                        )
                redacted_evidence.append(item)

            if any_evidence_redacted:
                result["evidence"] = redacted_evidence

        if pii_types_found:
            result["pii_detected"] = sorted(pii_types_found)

        return result
