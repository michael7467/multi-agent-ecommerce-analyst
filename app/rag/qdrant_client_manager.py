from __future__ import annotations

from qdrant_client import QdrantClient


_QDRANT_CLIENT: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    global _QDRANT_CLIENT

    if _QDRANT_CLIENT is None:
        _QDRANT_CLIENT = QdrantClient(path="artifacts/qdrant_storage")

    return _QDRANT_CLIENT