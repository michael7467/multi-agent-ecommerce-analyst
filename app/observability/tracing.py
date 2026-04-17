from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
except Exception:
    OTLPSpanExporter = None


_TRACING_INITIALIZED = False


def setup_tracing() -> None:
    global _TRACING_INITIALIZED

    if _TRACING_INITIALIZED:
        return

    service_name = os.getenv("OTEL_SERVICE_NAME", "multi-agent-ecommerce-analyst")
    exporter_mode = os.getenv("OTEL_TRACES_EXPORTER_MODE", "console").lower()

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "1.0.0",
        }
    )

    provider = TracerProvider(resource=resource)

    if exporter_mode == "otlp" and OTLPSpanExporter is not None:
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "http://localhost:4318/v1/traces",
        )
        processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
    else:
        processor = BatchSpanProcessor(ConsoleSpanExporter())

    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    _TRACING_INITIALIZED = True


def get_tracer(name: str = "app.tracing"):
    setup_tracing()
    return trace.get_tracer(name)