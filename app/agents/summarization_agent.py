from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent
from app.services.aspect_service import AspectService
from app.services.summarization_service import SummarizationService

logger = get_logger("agents.summarization")


class SummarizationAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="SummarizationAgent")
        self.aspect_service = AspectService()
        self.summarization_service = SummarizationService()

    @traced_agent("SummarizationAgent.run")
    def run(self, product_id: str, top_k: int = 3) -> dict:
        if not isinstance(product_id, str) or not product_id.strip():
            raise ValueError("SummarizationAgent: product_id must be a non-empty string")

        aspect_evidence = self.aspect_service.get_aspect_evidence(
            product_id=product_id,
            top_k=top_k,
        )

        aspect_summaries = {}
        for aspect, evidence in aspect_evidence.items():
            # Each aspect is its own LLM call. Previously, one failing
            # aspect (rate limit, transient error, malformed response)
            # discarded every summary already produced in this loop -- the
            # same all-or-nothing pattern the orchestrator's
            # critical/non-critical split exists to avoid, just recurring
            # here at the level of a single agent's internal loop instead
            # of across agents.
            try:
                summary = self.summarization_service.summarize_aspect(
                    product_id=product_id,
                    aspect=aspect,
                    evidence=evidence,
                )
                aspect_summaries[aspect] = {
                    "summary": summary,
                    "evidence": evidence,
                }
            except Exception:
                logger.error(
                    f"{self.name}: failed to summarize aspect '{aspect}'",
                    exc_info=True,
                )
                # Marked distinctly rather than omitted -- summarize_aspect
                # already returns a real, plain-text "no evidence found"
                # message for the genuinely-no-evidence case, so a missing
                # or None summary here needs to read differently from
                # that, not look like the same thing.
                aspect_summaries[aspect] = {
                    "summary": None,
                    "evidence": evidence,
                    "summary_failed": True,
                }

        return {
            "aspect_summaries": aspect_summaries
        }


if __name__ == "__main__":
    agent = SummarizationAgent()
    result = agent.run(product_id="B09SPZPDJK", top_k=2)

    for aspect, item in result["aspect_summaries"].items():
        print(f"\n=== {aspect.upper()} ===")
        print(item["summary"])