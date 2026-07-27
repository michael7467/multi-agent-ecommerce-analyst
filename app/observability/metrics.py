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

# Was previously flying entirely blind: agent_errors_total counts that
# something failed, but not what kind of failure. ValueError is the
# convention this codebase uses throughout for input validation failures
# (e.g. "product_id must be a non-empty string") -- distinct from a
# genuine infrastructure/dependency failure, and worth being able to tell
# apart in a dashboard rather than lumped into one generic error count.
agent_validation_failures_total = Counter(
    "agent_validation_failures_total",
    "Total number of agent input validation failures (ValueError specifically)",
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
# -----------------------------
# LLM Reliability Metrics
# -----------------------------
# Neither of these existed before, despite this codebase having several
# places that already carefully distinguish these exact outcomes in code
# (the "method": "llm_fallback" tags in aspect_sentiment_service.py, the
# exception-vs-unparseable split in every LLM-judge built this session) --
# that distinction was never surfaced anywhere a dashboard could see it.
parse_failure_total = Counter(
    "parse_failure_total",
    "Total number of LLM response parsing failures (malformed/unparseable output)",
    ["component"],  # e.g., llm_relevance_judge, faithfulness_judge, critic_agent, planning_agent
    registry=registry,
)

fallback_usage_total = Counter(
    "fallback_usage_total",
    "Total number of times a fallback/degraded result was used instead of a genuine one",
    ["component", "reason"],  # reason e.g., llm_call_failed, unparseable_response, contradiction_detected
    registry=registry,
)

# -----------------------------
# Query Classification Metrics
# -----------------------------
# Nothing tracked which kind of query the planner was actually seeing --
# needed for a "query type distribution" dashboard panel, which can't be
# derived from any other existing metric.
# -----------------------------
# Generation Reliability Metrics
# -----------------------------
# Derived from CriticAgent's existing hallucination_risk score, not a new
# LLM call -- see app/evaluation/metrics/generation_metrics.py, which
# already has the 1-10 -> 0-1 conversion this reuses. This is the live,
# every-request version; FaithfulnessJudge's per-claim check (built for
# the generation eval turn) is deliberately NOT wired here -- it's an
# expensive, ground-truth-dependent check meant for periodic offline
# evaluation, not something to run on every live request.
critic_hallucination_rate = Histogram(
    "critic_hallucination_rate",
    "Hallucination rate derived from CriticAgent's hallucination_risk score (0=none, 1=high)",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=registry,
)

query_type_total = Counter(
    "query_type_total",
    "Total number of queries matching each intent category in planning_agent's rule set",
    ["query_type"],  # e.g., buy_decision, opinion, aspect, pricing, unmatched
    registry=registry,
)

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