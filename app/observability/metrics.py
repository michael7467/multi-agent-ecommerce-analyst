from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server


_METRICS_STARTED = False


# Total analysis requests
ANALYSIS_REQUESTS_TOTAL = Counter(
    "analysis_requests_total",
    "Total number of analysis requests",
)

# Total analysis errors
ANALYSIS_ERRORS_TOTAL = Counter(
    "analysis_errors_total",
    "Total number of analysis errors",
)

# Cache hits / misses
CACHE_HITS_TOTAL = Counter(
    "cache_hits_total",
    "Total number of cache hits",
)

CACHE_MISSES_TOTAL = Counter(
    "cache_misses_total",
    "Total number of cache misses",
)

# Total retrieval requests
RETRIEVAL_REQUESTS_TOTAL = Counter(
    "retrieval_requests_total",
    "Total number of retrieval requests",
)

# Analysis latency
ANALYSIS_LATENCY_SECONDS = Histogram(
    "analysis_latency_seconds",
    "End-to-end analysis latency in seconds",
)

# Retrieval latency
RETRIEVAL_LATENCY_SECONDS = Histogram(
    "retrieval_latency_seconds",
    "Vector retrieval latency in seconds",
)

# LLM report latency
REPORT_LATENCY_SECONDS = Histogram(
    "report_latency_seconds",
    "LLM report generation latency in seconds",
)

# Active analysis requests
IN_PROGRESS_ANALYSIS = Gauge(
    "in_progress_analysis",
    "Number of analysis requests currently in progress",
)


def setup_metrics(port: int = 8001) -> None:
    global _METRICS_STARTED

    if _METRICS_STARTED:
        return

    start_http_server(port)
    _METRICS_STARTED = True