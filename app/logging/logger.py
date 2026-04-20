import logging
import os
import sys
from datetime import datetime
from pythonjsonlogger import jsonlogger

from opentelemetry.trace import get_current_span

def get_trace_id():
    span = get_current_span()
    if span and span.get_span_context().trace_id:
        trace_id = format(span.get_span_context().trace_id, "032x")
        return trace_id
    return None

def _get_log_level() -> str:
    return os.getenv("LOG_LEVEL", "INFO").upper()


def _get_log_file_path() -> str:
    base_dir = os.getcwd()
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    return os.path.join(log_dir, log_file)


def configure_logger(service_name: str = "multi-agent-ecommerce-analyst") -> logging.Logger:
    """
    Production-ready logger:
    - JSON logs
    - STDOUT handler (Kubernetes)
    - Optional file handler (local dev only)
    - Environment-based log level
    """
    logger = logging.getLogger(service_name)
    logger.setLevel(_get_log_level())
    logger.handlers.clear()
    logger.propagate = False

    json_formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(lineno)d %(message)s %(trace_id)s"
    )

    # Always log to STDOUT (Kubernetes best practice)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(json_formatter)
    logger.addHandler(stdout_handler)

    # Enable file logging ONLY if explicitly set
    if os.getenv("ENABLE_FILE_LOGS", "false").lower() == "true":
        try:
            file_handler = logging.FileHandler(_get_log_file_path())
            file_handler.setFormatter(json_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.error("Failed to initialize file logging", extra={"error": str(e)})

    logger.info("Logger initialized", extra={"service": service_name})
    return logger



# Convenience function for modules that just want a logger
class TraceIdAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        trace_id = get_trace_id()
        kwargs["extra"] = kwargs.get("extra", {})
        kwargs["extra"]["trace_id"] = trace_id
        return msg, kwargs

def get_logger(name: str):
    base_logger = configure_logger()
    return TraceIdAdapter(base_logger.getChild(name), {})
