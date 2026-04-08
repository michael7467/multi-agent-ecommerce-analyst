from fastapi import APIRouter, HTTPException

from app.agents.orchestrator import Orchestrator
from app.api.schemas.request import AnalyzeProductRequest
from app.api.schemas.response import AnalyzeProductResponse

router = APIRouter()
orchestrator = Orchestrator()


@router.post("/analyze-product", response_model=AnalyzeProductResponse)
def analyze_product(request: AnalyzeProductRequest):
    try:
        result = orchestrator.run(
            product_id=request.product_id,
            query=request.query,
            top_k=request.top_k,
        )
        return result["final_output"]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))