from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_orchestrator
from app.api.errors import APIError
from app.api.schemas.analysis import AnalyzeRequest, AnalyzeResponse
from app.agents.dynamic_orchestrator import DynamicOrchestrator


router = APIRouter(tags=["analysis"])


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_product(
    payload: AnalyzeRequest,
    orchestrator: DynamicOrchestrator = Depends(get_orchestrator),
) -> AnalyzeResponse:
    try:
        result = orchestrator.run(
            product_id=payload.product_id,
            query=payload.query,
            top_k=payload.top_k,
        )
        return AnalyzeResponse(**result)
    except ValueError as exc:
        raise APIError(str(exc), status_code=400) from exc