from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import APIError, api_error_handler, generic_error_handler
from app.api.middleware.auth_middleware import APIKeyMiddleware
from app.api.middleware.rate_limit_middleware import RateLimitMiddleware
from app.api.middleware.request_logging import RequestLoggingMiddleware
from app.api.routes.analysis import router as analysis_router
from app.api.routes.health import router as health_router
from app.config.settings import settings
from app.logging.logger import get_logger
from app.memory.db import init_db
from app.observability.metrics import metrics_router
from app.observability.tracing import setup_tracing

logger = get_logger("api.main")

setup_tracing()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API service starting up")

    try:
        init_db()
        logger.info("Memory database initialized")
    except Exception:
        logger.error("Failed to initialize memory database", exc_info=True)
        raise

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(APIError, api_error_handler)
app.add_exception_handler(Exception, generic_error_handler)

app.include_router(health_router)
app.include_router(analysis_router)
app.include_router(metrics_router)


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
    }