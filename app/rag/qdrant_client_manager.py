from __future__ import annotations

from threading import Lock
from qdrant_client import QdrantClient

from app.config.settings import settings
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent

logger = get_logger("qdrant.client")

_QDRANT_CLIENT: QdrantClient | None = None
_QDRANT_LOCK = Lock()


@traced_agent("qdrant_build")
def get_qdrant_client() -> QdrantClient:
    global _QDRANT_CLIENT

    if _QDRANT_CLIENT is not None:
        return _QDRANT_CLIENT

    with _QDRANT_LOCK:
        if _QDRANT_CLIENT is not None:
            return _QDRANT_CLIENT

        if settings.qdrant_mode not in {"server", "local"}:
            raise ValueError("qdrant_mode must be 'server' or 'local'")

        try:
            if settings.qdrant_mode == "server":
                _QDRANT_CLIENT = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key or None,
                    check_compatibility=False,
                )
            else:
                _QDRANT_CLIENT = QdrantClient(
                    path=settings.qdrant_storage_path,
                )

            _QDRANT_CLIENT.get_collections()

        except Exception:
            logger.error("Failed to initialize Qdrant client", exc_info=True)
            raise

        logger.info(
            f"Qdrant client initialized in {settings.qdrant_mode} mode"
        )
        return _QDRANT_CLIENT