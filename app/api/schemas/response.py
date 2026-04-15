from pydantic import BaseModel
from typing import List, Optional

class RecommendationItem(BaseModel):
    product_id: str
    title: str
    categories: str
    price: float | None = None
    predicted_class: str
    similarity_score: float

class EvidenceItem(BaseModel):
    product_id: str
    title: str
    review_text: str
    review_title: str
    categories: str
    score: float


class AnalyzeProductResponse(BaseModel):
    product_id: str
    title: str
    categories: str
    price: Optional[float] = None
    predicted_class: str
    evidence: List[EvidenceItem]
    recommendations: List[RecommendationItem]
    report: str
    guardrail_status: str