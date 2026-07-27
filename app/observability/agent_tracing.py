from __future__ import annotations

import inspect
from functools import wraps

from opentelemetry import trace

from app.logging.logger import get_logger

logger = get_logger("agents.tracing")
tracer = trace.get_tracer("multi-agent-ecommerce-analyst")


def _on_success(span, agent_name: str) -> None:
    span.set_attribute("agent.success", True)
    logger.info(f"{agent_name} completed")


def _on_failure(span, agent_name: str, exc: Exception) -> None:
    # start_as_current_span already records the exception and sets the
    # span status to ERROR automatically once this exception propagates
    # out of the `with` block below (confirmed by testing, not assumed) --
    # calling span.record_exception() here too just double-records the
    # same event.
    span.set_attribute("agent.success", False)
    logger.error(f"{agent_name} failed", exc_info=True)


def traced_agent(agent_name: str):
    """
    Decorator to trace agent execution as its own span.

    This logging is only redundant for agents the orchestrator calls
    directly through _safe_agent_call, which logs the same event with more
    context (trace_id, input_args). Several agents are invoked directly by
    other services rather than by the orchestrator (e.g. the sub-agents
    CompetitiveService calls internally) -- for those, this is the only
    log record of success or failure, so it stays.
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(agent_name) as span:
                span.set_attribute("agent.name", agent_name)
                logger.info(f"{agent_name} started")
                try:
                    result = await func(*args, **kwargs)
                    _on_success(span, agent_name)
                    return result
                except Exception as exc:
                    _on_failure(span, agent_name, exc)
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(agent_name) as span:
                span.set_attribute("agent.name", agent_name)
                logger.info(f"{agent_name} started")
                try:
                    result = func(*args, **kwargs)
                    _on_success(span, agent_name)
                    return result
                except Exception as exc:
                    _on_failure(span, agent_name, exc)
                    raise

        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

    return decorator