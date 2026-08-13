from __future__ import annotations

from contextlib import asynccontextmanager, AsyncExitStack

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.transport_security import TransportSecuritySettings

from app.api.errors import APIError, api_error_handler, generic_error_handler
from app.api.middleware.auth_middleware import APIKeyMiddleware
from app.api.middleware.rate_limit_middleware import RateLimitMiddleware
from app.api.middleware.request_logging import RequestLoggingMiddleware
from app.api.routes.analysis import router as analysis_router
from app.api.routes.health import router as health_router
from app.api.routes.admin import router as admin_router
from app.config.settings import settings
from app.logging.logger import get_logger
from app.mcp.server import mcp_server
from app.memory.db import init_db
from app.observability.metrics import metrics_router
from app.observability.tracing import setup_tracing

logger = get_logger("api.main")

setup_tracing()

# streamable_http_path='/' avoids the double-nested /mcp/mcp path that
# resulted from mounting at /mcp externally while the app's own default
# internal path was also /mcp -- confirmed directly by testing, not
# assumed. allowed_hosts is read from settings rather than hardcoded,
# since the real deployment hostname isn't something fixed at build time
# -- verified during testing that omitting or misconfiguring this
# produces a 421 rejection, not a silent bypass.
mcp_app = mcp_server.streamable_http_app(
    streamable_http_path="/",
    transport_security=TransportSecuritySettings(
        allowed_hosts=settings.mcp_allowed_hosts_list,
    ),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API service starting up")

    try:
        init_db()
        logger.info("Memory database initialized")
    except Exception:
        logger.error("Failed to initialize memory database", exc_info=True)
        raise

    # The MCP app's session manager needs its own lifespan entered
    # explicitly -- confirmed directly by testing that mounting alone is
    # NOT sufficient; without this, every tool call fails with
    # "Task group is not initialized."
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(mcp_app.router.lifespan_context(mcp_app))
        logger.info("MCP server initialized")

        yield

    logger.info("API service shutting down")


app = FastAPI(
    title="Multi-Agent E-commerce AI Analyst API",
    version=settings.app_version,
    description="Production API for product analysis, retrieval, recommendation, and decision support.",
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(APIError, api_error_handler)
app.add_exception_handler(Exception, generic_error_handler)

app.include_router(health_router)
app.include_router(analysis_router)
app.include_router(admin_router)
app.include_router(metrics_router)

# Mounted last, deliberately: since app.mount() registers routes at
# import time, it needs everything above it (middleware, exception
# handlers) already in place. Protected by the same APIKeyMiddleware as
# everything else -- confirmed directly by testing that parent
# middleware applies to mounted sub-applications, not assumed. Any valid
# key works here, same as /analyze -- not admin-gated, since MCP calls
# are functionally equivalent to that endpoint, just over a different
# protocol, not a destructive or operational action.
app.mount("/mcp", mcp_app)


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
    }