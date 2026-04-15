from __future__ import annotations

from app.services.rag_service import RAGService


class AspectService:
    ASPECT_QUERIES = {
        "sound_quality": "sound quality audio bass treble volume hollow sound",
        "battery_life": "battery life charging power lasts long battery",
        "comfort": "comfort comfortable fit bulky wear earpads head",
        "build_quality": "build quality design material premium solid construction",
        "durability": "durability durable broke broken long lasting quality issue",
        "price_value": "price value worth money expensive cheap overpriced deal",
    }

    def __init__(self) -> None:
        self.rag_service = RAGService()

    def get_aspect_evidence(self, product_id: str, top_k: int = 3) -> dict[str, list[dict]]:
        aspect_results: dict[str, list[dict]] = {}

        for aspect, query in self.ASPECT_QUERIES.items():
            evidence = self.rag_service.get_product_evidence(
                product_id=product_id,
                query=query,
                top_k=top_k,
            )
            aspect_results[aspect] = evidence

        return aspect_results


if __name__ == "__main__":
    service = AspectService()
    result = service.get_aspect_evidence(product_id="B09SPZPDJK", top_k=2)

    for aspect, evidence in result.items():
        print(f"\n=== {aspect.upper()} ===")
        for item in evidence:
            print(item.get("review_title", ""), "->", item.get("score", 0))