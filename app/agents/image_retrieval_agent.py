from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.rag.image_retriever import ImageRetriever
from app.observability.agent_tracing import traced_agent

class ImageRetrievalAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="ImageRetrievalAgent")
        self.retriever = ImageRetriever()

    @traced_agent("ImageRetrievalAgent.run")
    def run(self, product_id: str, top_k: int = 5) -> dict:
        results_df = self.retriever.search_by_product(
            product_id=product_id,
            top_k=top_k,
        )

        if results_df is None or results_df.empty:
            return {"image_similar_products": []}

        return {
            "image_similar_products": results_df.to_dict(orient="records")
        }