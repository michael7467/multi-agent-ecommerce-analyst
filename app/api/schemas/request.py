from pydantic import BaseModel, Field


class AnalyzeProductRequest(BaseModel):
    product_id: str = Field(..., example="B09SPZPDJK")
    query: str = Field(..., example="sound quality and noise cancellation")
    top_k: int = Field(default=3, ge=1, le=10)