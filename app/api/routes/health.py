from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.api.schemas.analysis import HealthResponse
from app.cache.redis_client import get_redis_client
from app.logging.logger import get_logger
from opentelemetry.trace import get_current_span

router = APIRouter(tags=["health"])
logger = get_logger("api.health")


def _get_trace_id() -> str | None:
    span = get_current_span()
    if not span:
        return None
    ctx = span.get_span_context()
    if not ctx or ctx.trace_id == 0:
        return None
    return format(ctx.trace_id, "032x")


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """
    Liveness probe: returns OK if the API process is running.
    Should NOT check external dependencies.
    """
    trace_id = _get_trace_id()
    correlation_id = getattr(request.state, "correlation_id", None)

    logger.info(
        "Liveness check",
        extra={"trace_id": trace_id, "correlation_id": correlation_id},
    )

    return HealthResponse(status="ok", service="multi-agent-ecommerce-analyst-api")


@router.get("/ready", response_model=HealthResponse)
def ready(request: Request) -> JSONResponse | HealthResponse:
    """
    Readiness probe: checks external dependencies.
    If Redis (or future dependencies) are down, return 503.
    """
    trace_id = _get_trace_id()
    correlation_id = getattr(request.state, "correlation_id", None)

    # Check Redis
    client = get_redis_client()
    if client is not None:
        try:
            client.ping()
        except Exception as exc:
            logger.error(
                "Redis readiness check failed",
                extra={
                    "error": str(exc),
                    "trace_id": trace_id,
                    "correlation_id": correlation_id,
                },
            )
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "type": "dependency_unavailable",
                        "message": "Redis not ready",
                        "trace_id": trace_id,
                        "correlation_id": correlation_id,
                    }
                },
            )

    logger.info(
        "Readiness check passed",
        extra={"trace_id": trace_id, "correlation_id": correlation_id},
    )

    return HealthResponse(status="ready", service="multi-agent-ecommerce-analyst-api")
