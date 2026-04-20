from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import APIError, api_error_handler, generic_error_handler
from app.api.routes.analysis import router as analysis_router
from app.api.routes.health import router as health_router
from app.observability.metrics import setup_metrics
from app.observability.tracing import setup_tracing
from app.core.config import settings


setup_tracing()
setup_metrics(port=settings.metrics_port)

app = FastAPI(
    title="Multi-Agent E-commerce AI Analyst API",
    version="1.0.0",
    description="Production API for product analysis, retrieval, recommendation, and decision support.",
)

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