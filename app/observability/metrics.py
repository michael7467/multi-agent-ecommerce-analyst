from __future__ import annotations

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from fastapi import APIRouter, Response

# Shared registry for all metrics
registry = CollectorRegistry()

# -----------------------------
# Agent Metrics
# -----------------------------
agent_execution_seconds = Histogram(
    "agent_execution_seconds",
    "Execution time of each agent",
    ["agent"],
    registry=registry,
)

agent_errors_total = Counter(
    "agent_errors_total",
    "Total number of agent errors",
    ["agent"],
    registry=registry,
)

# -----------------------------
# Analysis Metrics
# -----------------------------
ANALYSIS_REQUESTS_TOTAL = Counter(
    "analysis_requests_total",
    "Total number of analysis requests",
    ["endpoint"],  # Improvement: label support
    registry=registry,
)

ANALYSIS_ERRORS_TOTAL = Counter(
    "analysis_errors_total",
    "Total number of analysis errors",
    ["endpoint"],  # Improvement: label support
    registry=registry,
)

ANALYSIS_LATENCY_SECONDS = Histogram(
    "analysis_latency_seconds",
    "End-to-end analysis latency in seconds",
    ["endpoint"],  # Improvement: label support
    registry=registry,
)

IN_PROGRESS_ANALYSIS = Gauge(
    "in_progress_analysis",
    "Number of analysis requests currently in progress",
    registry=registry,
)

# -----------------------------
# Retrieval Metrics
# -----------------------------
RETRIEVAL_REQUESTS_TOTAL = Counter(
    "retrieval_requests_total",
    "Total number of retrieval requests",
    ["source"],  # e.g., qdrant, cache, hybrid
    registry=registry,
)

RETRIEVAL_LATENCY_SECONDS = Histogram(
    "retrieval_latency_seconds",
    "Vector retrieval latency in seconds",
    ["source"],
    registry=registry,
)

# -----------------------------
# Cache Metrics
# -----------------------------
CACHE_HITS_TOTAL = Counter(
    "cache_hits_total",
    "Total number of cache hits",
    ["cache_name"],  # e.g., embedding_cache, product_cache
    registry=registry,
)

CACHE_MISSES_TOTAL = Counter(
    "cache_misses_total",
    "Total number of cache misses",
    ["cache_name"],
    registry=registry,
)

# -----------------------------
# LLM Metrics
# -----------------------------
REPORT_LATENCY_SECONDS = Histogram(
    "report_latency_seconds",
    "LLM report generation latency in seconds",
    ["model"],  # e.g., gpt-4, llama3-70b
    registry=registry,
)

# -----------------------------
# FastAPI Metrics Endpoint
# -----------------------------
metrics_router = APIRouter()

@metrics_router.get("/metrics")
def metrics():
    """
    Exposes Prometheus metrics at /metrics.
    This replaces start_http_server() and integrates cleanly with FastAPI.
    """
    data = generate_latest(registry)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
