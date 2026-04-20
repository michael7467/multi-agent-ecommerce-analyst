from __future__ import annotations

import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from redis import Redis, RedisError
from app.logging.logger import get_logger
from app.config.settings import settings
from app.observability.tracing import get_tracer

logger = get_logger("api.rate_limit")
tracer = get_tracer("api.rate_limit")

redis = Redis.from_url(settings.redis_url, decode_responses=True)


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip health + metrics + CORS preflight
        if (
            path.startswith("/health")
            or path == "/metrics"
            or request.method == "OPTIONS"
        ):
            return await call_next(request)

        # Identify client (prefer API key over IP)
        api_key = request.headers.get("X-API-Key")
        client_id = api_key if api_key else request.client.host

        # Per-route overrides (optional)
        route_limit = settings.rate_limit_overrides.get(path)
        limit = route_limit or settings.rate_limit_per_minute

        key = f"rate_limit:{client_id}"

        try:
            # Increment request count atomically
            current = redis.incr(key)

            # Set TTL only on first increment
            if current == 1:
                redis.expire(key, 60)

            if current > limit:
                with tracer.start_as_current_span("rate_limit_exceeded") as span:
                    span.set_attribute("client.id", client_id)
                    span.set_attribute("http.path", path)
                    span.set_attribute("rate.limit", limit)

                logger.warning(
                    "Rate limit exceeded",
                    extra={
                        "client": client_id,
                        "path": path,
                        "limit": limit,
                    },
                )

                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "rate_limit_exceeded",
                        "detail": f"Rate limit of {limit} requests/min exceeded",
                    },
                )

        except RedisError as e:
            # Fail open: allow traffic if Redis is down
            logger.error("Redis unavailable for rate limiting", extra={"error": str(e)})
            return await call_next(request)

        return await call_next(request)
