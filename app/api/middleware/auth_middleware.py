from __future__ import annotations

import hmac
from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.logging.logger import get_logger
from app.config.settings import settings
from app.observability.tracing import get_tracer

logger = get_logger("api.security")
tracer = get_tracer("api.security")


def _parse_keys(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path


        if path.startswith("/health") or path == "/ready":
            return await call_next(request)

        # Allow CORS preflight
        if request.method == "OPTIONS":
            return await call_next(request)


        if not settings.api_key:
            request.state.role = "admin"
            return await call_next(request)


        auth_header = request.headers.get("Authorization", "")
        bearer_token = auth_header.removeprefix("Bearer ").strip() if auth_header.startswith("Bearer ") else None

        provided_key = (
            request.headers.get("X-API-Key")
            or request.query_params.get("api_key")
            or bearer_token
        )

 
        analyst_keys = _parse_keys(settings.api_key)
        admin_keys = _parse_keys(getattr(settings, "admin_api_keys", None))

        is_admin = any(
            hmac.compare_digest(provided_key or "", key) for key in admin_keys
        )
        is_analyst = any(
            hmac.compare_digest(provided_key or "", key) for key in analyst_keys
        )
        authorized = is_admin or is_analyst

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

   
        if path == "/metrics" and not is_admin:
            return JSONResponse(
                status_code=403,
                content={
                    "error": "forbidden",
                    "detail": "This endpoint requires an admin API key.",
                },
            )


        request.state.role = "admin" if is_admin else "analyst"

        return await call_next(request)


def require_admin(request: Request) -> None:

    role = getattr(request.state, "role", None)
    if role != "admin":
        raise HTTPException(
            status_code=403,
            detail="This action requires an admin API key.",
        )