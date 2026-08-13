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

    span.set_attribute("agent.success", False)
    logger.error(f"{agent_name} failed", exc_info=True)


def traced_agent(agent_name: str):

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