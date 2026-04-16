from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.rag.image_retriever import ImageRetriever


class ImageRetrievalAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="ImageRetrievalAgent")
        self.retriever = ImageRetriever()

    def run(self, product_id: str, top_k: int = 5) -> dict:
        results_df = self.retriever.search_by_product(
            product_id=product_id,
            top_k=top_k,
        )

        return {
            "image_similar_products": results_df.to_dict(orient="records")
        }


if __name__ == "__main__":
    agent = ImageRetrievalAgent()
    result = agent.run(product_id="B09SPZPDJK", top_k=3)

    print("\n=== IMAGE RETRIEVAL AGENT OUTPUT ===\n")
    print(result)