from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from opentelemetry.trace import get_current_span

from app.api.dependencies import get_orchestrator
from app.api.errors import APIError
from app.api.schemas.analysis import AnalyzeRequest, AnalyzeResponse
from app.agents.dynamic_orchestrator import DynamicOrchestrator
from app.logging.logger import get_logger

router = APIRouter(tags=["analysis"])
logger = get_logger("api.analysis")


def _get_trace_id() -> str | None:
    span = get_current_span()
    if not span:
        return None
    ctx = span.get_span_context()
    if not ctx or ctx.trace_id == 0:
        return None
    return format(ctx.trace_id, "032x")


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_product(
    payload: AnalyzeRequest,
    request: Request,
    orchestrator: DynamicOrchestrator = Depends(get_orchestrator),
) -> AnalyzeResponse:

    trace_id = _get_trace_id()
    correlation_id = getattr(request.state, "correlation_id", None)

    logger.info(
        "Analysis request received",
        extra={
            "product_id": payload.product_id,
            "query": payload.query,
            "top_k": payload.top_k,
            "trace_id": trace_id,
            "correlation_id": correlation_id,
        },
    )

    try:
        result = orchestrator.run(
            product_id=payload.product_id,
            query=payload.query,
            top_k=payload.top_k,
        )

        # Validate orchestrator output
        try:
            response = AnalyzeResponse(**result)
        except Exception as exc:
            logger.error(
                "Invalid orchestrator response",
                extra={
                    "result": result,
                    "trace_id": trace_id,
                    "correlation_id": correlation_id,
                },
            )
            raise APIError("Orchestrator returned invalid response", status_code=500) from exc

        logger.info(
            "Analysis completed successfully",
            extra={
                "product_id": payload.product_id,
                "trace_id": trace_id,
                "correlation_id": correlation_id,
            },
        )

        return response

    except APIError:
        raise

    except Exception as exc:
        logger.error(
            "Analysis failed",
            extra={
                "error": str(exc),
                "product_id": payload.product_id,
                "trace_id": trace_id,
                "correlation_id": correlation_id,
            },
        )
        raise APIError("Failed to analyze product", status_code=500) from exc
