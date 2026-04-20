from __future__ import annotations

import os
from app.config.settings import settings
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, SpanLimits
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
except Exception:
    OTLPSpanExporter = None

_TRACING_INITIALIZED = False

def setup_tracing() -> None:
    global _TRACING_INITIALIZED
    if _TRACING_INITIALIZED:
        return

    resource = Resource.create({
        "service.name": settings.otel_service_name,
        "service.version": "1.0.0",
        "deployment.environment": settings.env,
        "k8s.namespace": os.getenv("K8S_NAMESPACE", "default"),
    })

    provider = TracerProvider(
        resource=resource,
        sampler=TraceIdRatioBased(settings.tracing_sample_rate),
        span_limits=SpanLimits(
            max_attributes=128,
            max_events=128,
            max_links=32,
        )
    )

    if settings.otel_traces_exporter_mode == "otlp" and OTLPSpanExporter:
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
        processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    else:
        processor = BatchSpanProcessor(ConsoleSpanExporter())

    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    _TRACING_INITIALIZED = True

def get_tracer(name: str = "app.tracing"):
    setup_tracing()
    return trace.get_tracer(name)
