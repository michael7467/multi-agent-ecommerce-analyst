from __future__ import annotations

import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.logging.logger import get_logger
from opentelemetry.trace import get_current_span

logger = get_logger("api.request")


def safe_trace_id() -> str | None:
    """
    Safely extract the current OpenTelemetry trace ID.
    Returns None if no valid trace is active.
    """
    span = get_current_span()
    if not span:
        return None

    ctx = span.get_span_context()
    if not ctx or not ctx.trace_id or ctx.trace_id == 0:
        return None

    return format(ctx.trace_id, "032x")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip CORS preflight
        if request.method == "OPTIONS":
            return await call_next(request)

        start_time = time.perf_counter()

        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id

        trace_id = safe_trace_id()

        logger.info(
            "Incoming request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host,
                "correlation_id": correlation_id,
                "trace_id": trace_id,
            },
        )

        try:
            response: Response = await call_next(request)
        except Exception as e:
            logger.error(
                "Unhandled exception during request",
                extra={
                    "error": str(e),
                    "method": request.method,
                    "path": request.url.path,
                    "correlation_id": correlation_id,
                    "trace_id": safe_trace_id(),
                },
            )
            raise

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            "Outgoing response",
            extra={
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "method": request.method,
                "path": request.url.path,
                "correlation_id": correlation_id,
                "trace_id": safe_trace_id(),
            },
        )

        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id

        return response
