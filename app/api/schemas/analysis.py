from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    product_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class AnalyzeResponse(BaseModel):
    plan: dict[str, bool]
    final_output: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    service: str