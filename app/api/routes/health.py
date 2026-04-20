from fastapi import APIRouter, HTTPException

from app.api.schemas.analysis import HealthResponse
from app.cache.redis_client import get_redis_client

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="multi-agent-ecommerce-analyst-api")


@router.get("/ready", response_model=HealthResponse)
def ready() -> HealthResponse:
    client = get_redis_client()

    if client is not None:
        try:
            client.ping()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Redis not ready") from exc

    return HealthResponse(status="ready", service="multi-agent-ecommerce-analyst-api")