from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.rag.image_retriever import ImageRetriever
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent

logger = get_logger("agents.image_retrieval")

class ImageRetrievalAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="ImageRetrievalAgent")
        self.retriever = ImageRetriever()

    @traced_agent
    def run(self, product_id: str, top_k: int = 5) -> dict:
        if not isinstance(product_id, str):
            raise ValueError("ImageRetrievalAgent: product_id must be a string")

        try:
            results_df = self.retriever.search_by_product(
                product_id=product_id,
                top_k=top_k,
            )
        except Exception:
            logger.error(f"{self.name}: image retrieval failed", exc_info=True)
            raise

        if results_df is None or results_df.empty:
            return {"image_similar_products": []}

        return {
            "image_similar_products": results_df.to_dict(orient="records")
        }
