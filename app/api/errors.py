from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.logging.logger import get_logger
from opentelemetry.trace import get_current_span

logger = get_logger("api.errors")


def _get_trace_id() -> str | None:
    span = get_current_span()
    if not span:
        return None
    ctx = span.get_span_context()
    if not ctx or ctx.trace_id == 0:
        return None
    return format(ctx.trace_id, "032x")


def _error_response(
    status_code: int,
    error_type: str,
    message: str,
    request: Request,
) -> JSONResponse:
    trace_id = _get_trace_id()
    correlation_id = getattr(request.state, "correlation_id", None)

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "type": error_type,
                "message": message,
                "status_code": status_code,
                "trace_id": trace_id,
                "correlation_id": correlation_id,
            }
        },
    )


class APIError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    logger.warning(
        "APIError raised",
        extra={
            "message": exc.message,
            "status_code": exc.status_code,
            "path": request.url.path,
            "trace_id": _get_trace_id(),
            "correlation_id": getattr(request.state, "correlation_id", None),
        },
    )
    return _error_response(exc.status_code, "api_error", exc.message, request)


async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    logger.warning(
        "HTTPException raised",
        extra={
            "detail": exc.detail,
            "status_code": exc.status_code,
            "path": request.url.path,
            "trace_id": _get_trace_id(),
            "correlation_id": getattr(request.state, "correlation_id", None),
        },
    )
    return _error_response(exc.status_code, "http_error", str(exc.detail), request)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning(
        "Validation error",
        extra={
            "errors": exc.errors(),
            "path": request.url.path,
            "trace_id": _get_trace_id(),
            "correlation_id": getattr(request.state, "correlation_id", None),
        },
    )
    return _error_response(422, "validation_error", "Invalid request payload", request)


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled exception",
        extra={
            "error": str(exc),
            "path": request.url.path,
            "trace_id": _get_trace_id(),
            "correlation_id": getattr(request.state, "correlation_id", None),
        },
    )
    return _error_response(500, "internal_server_error", "An unexpected error occurred.", request)
