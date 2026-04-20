from functools import wraps
from opentelemetry import trace
from app.logging.logger import get_logger

logger = get_logger("agents.tracing")
tracer = trace.get_tracer("multi-agent-ecommerce-analyst")

def traced_agent(agent_name: str):
    """
    Decorator to trace agent execution as its own span.
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(agent_name) as span:
                try:
                    logger.info(f"{agent_name} started")
                    result = await func(*args, **kwargs)

                    # Add metadata to span
                    span.set_attribute("agent.name", agent_name)
                    span.set_attribute("agent.success", True)

                    logger.info(f"{agent_name} completed")
                    return result

                except Exception as e:
                    span.set_attribute("agent.success", False)
                    span.record_exception(e)
                    logger.error(f"{agent_name} failed", extra={"error": str(e)})
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(agent_name) as span:
                try:
                    logger.info(f"{agent_name} started")
                    result = func(*args, **kwargs)

                    span.set_attribute("agent.name", agent_name)
                    span.set_attribute("agent.success", True)

                    logger.info(f"{agent_name} completed")
                    return result

                except Exception as e:
                    span.set_attribute("agent.success", False)
                    span.record_exception(e)
                    logger.error(f"{agent_name} failed", extra={"error": str(e)})
                    raise

        return async_wrapper if func.__code__.co_flags & 0x80 else sync_wrapper

    return decorator
