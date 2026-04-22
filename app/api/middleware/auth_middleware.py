from __future__ import annotations

import hmac
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.logging.logger import get_logger
from app.config.settings import settings
from app.observability.tracing import get_tracer

logger = get_logger("api.security")
tracer = get_tracer("api.security")


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Allow health, readiness, and metrics without auth
        if path.startswith("/health") or path == "/ready" or path == "/metrics":
            return await call_next(request)

        # Allow CORS preflight
        if request.method == "OPTIONS":
            return await call_next(request)

        # If auth is disabled, allow all requests
        if not settings.api_key:
            return await call_next(request)

        # Extract API key (header or query param)
        provided_key = (
            request.headers.get("X-API-Key")
            or request.query_params.get("api_key")
        )

        # Support multiple API keys (comma-separated in env)
        valid_keys = [k.strip() for k in settings.api_key.split(",") if k.strip()]

        # Constant-time comparison
        authorized = any(
            hmac.compare_digest(provided_key or "", valid_key)
            for valid_key in valid_keys
        )

        if not authorized:
            with tracer.start_as_current_span("unauthorized_request") as span:
                span.set_attribute("security.unauthorized", True)
                span.set_attribute("http.path", path)
                span.set_attribute("client.ip", request.client.host if request.client else "unknown")

                logger.warning(
                    "Unauthorized request",
                    extra={
                        "path": path,
                        "client": request.client.host if request.client else "unknown",
                        "trace_id": span.get_span_context().trace_id,
                    },
                )

            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "detail": "Invalid or missing API key",
                },
            )

        return await call_next(request)