from app.core.config import settings
from qdrant_client import QdrantClient


_QDRANT_CLIENT: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    global _QDRANT_CLIENT

    if _QDRANT_CLIENT is not None:
        return _QDRANT_CLIENT

    if settings.qdrant_mode == "server":
        _QDRANT_CLIENT = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
    else:
        _QDRANT_CLIENT = QdrantClient(
            path=settings.qdrant_storage_path,
        )

    return _QDRANT_CLIENT