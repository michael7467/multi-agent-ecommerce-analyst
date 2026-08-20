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