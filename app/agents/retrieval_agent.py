from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.services.rag_service import RAGService


class RetrievalAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="RetrievalAgent")
        self.rag_service = RAGService()

    def run(self, product_id: str, query: str, top_k: int = 3) -> dict:
        evidence = self.rag_service.get_product_evidence(
            product_id=product_id,
            query=query,
            top_k=top_k,
        )
        return {
            "evidence": evidence
        }