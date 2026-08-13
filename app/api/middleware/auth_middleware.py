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

        # Allow health and readiness without auth. Metrics used to be
        # exempt here too -- now it goes through the normal key-check
        # flow below like any other route, so a request with no key at
        # all still gets a clean 401 ("who are you") rather than being
        # silently let through or confusingly rejected before even
        # reaching the authorization logic.
        if path.startswith("/health") or path == "/ready":
            return await call_next(request)

        # Allow CORS preflight
        if request.method == "OPTIONS":
            return await call_next(request)

        # If auth is disabled, allow all requests -- and since this is
        # already a "trust everything" dev/test posture, role checks
        # would just be theater on top of it. Everyone gets admin so
        # downstream role-gated routes don't additionally block local
        # development that already opted out of auth entirely.
        if not settings.api_key:
            request.state.role = "admin"
            return await call_next(request)

        # Extract API key (header, query param, or standard Authorization:
        # Bearer -- the last one specifically so Prometheus's native
        # bearer_token scrape config works. Prometheus doesn't support
        # arbitrary custom headers like X-API-Key in scrape configs at
        # all (confirmed against open issues on prometheus/prometheus
        # itself, not assumed) -- Authorization: Bearer is the one
        # mechanism it natively, reliably sends.
        auth_header = request.headers.get("Authorization", "")
        bearer_token = auth_header.removeprefix("Bearer ").strip() if auth_header.startswith("Bearer ") else None

        provided_key = (
            request.headers.get("X-API-Key")
            or request.query_params.get("api_key")
            or bearer_token
        )

        # Two separate key lists now, not one -- admin keys are a new,
        # additive settings field (admin_api_keys). The original api_key
        # field and its comma-separated multi-key behavior are completely
        # unchanged; it's just being read here as the "analyst" tier
        # specifically, rather than an undifferentiated "valid" list.
        analyst_keys = _parse_keys(settings.api_key)
        admin_keys = _parse_keys(getattr(settings, "admin_api_keys", None))

        # Constant-time comparison, same as before, now against both lists
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

        # /metrics specifically requires admin, not just any valid key --
        # checked after the general 401 check above, so a request with a
        # genuinely wrong/missing key still gets 401 ("who are you"), and
        # only a validly-authenticated-but-wrong-role request gets this
        # 403 ("I know who you are, you still can't see this").
        if path == "/metrics" and not is_admin:
            return JSONResponse(
                status_code=403,
                content={
                    "error": "forbidden",
                    "detail": "This endpoint requires an admin API key.",
                },
            )

        # An admin key is valid for everything an analyst key is, plus
        # more -- so a key that matched admin_keys is tagged "admin", not
        # forced to also appear in the analyst list to be let in at all.
        request.state.role = "admin" if is_admin else "analyst"

        return await call_next(request)


def require_admin(request: Request) -> None:
    """
    FastAPI dependency for gating specific routes to admin-role requests
    only. Use via Depends(require_admin) on any route that changes system
    state rather than just reading from it.

    Reads request.state.role, which APIKeyMiddleware sets on every
    authorized request above -- this dependency only enforces the check,
    it doesn't duplicate the key-validation logic itself.
    """
    role = getattr(request.state, "role", None)
    if role != "admin":
        raise HTTPException(
            status_code=403,
            detail="This action requires an admin API key.",
        )