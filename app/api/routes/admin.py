from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.middleware.auth_middleware import require_admin
from app.rag.qdrant_index_builder import QdrantIndexBuilder
from app.logging.logger import get_logger

logger = get_logger("api.admin")

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/rebuild-index", dependencies=[Depends(require_admin)])
def rebuild_index() -> dict:
    """
    Full delete-and-recreate of the Qdrant collection from the current
    embeddings/metadata files on disk.

    Synchronous -- the request blocks until the rebuild completes. That's
    a deliberate, proportionate choice given this project's actual scale
    (a few thousand points, not millions); if the collection ever grows
    large enough to risk a real HTTP timeout, the next step would be
    moving this to a background task with a separate status endpoint,
    not something needed at the current scale.

    self.client here is the same shared, module-level Qdrant client
    singleton every other agent in this process reads from via
    get_qdrant_client() -- QdrantIndexBuilder.close() is deliberately
    NEVER called in this endpoint. Closing it here would silently break
    every other agent's Qdrant access for the rest of this process's
    life, since the singleton cache would still hold a reference to a
    now-closed client with nothing to recreate it. That's exactly the
    hazard flagged when this class was first reviewed, now the concrete
    scenario it was flagged for.
    """
    builder = QdrantIndexBuilder()

    try:
        builder.build()
    except Exception as exc:
        logger.error("Index rebuild failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Index rebuild failed: {exc}") from exc

    return {"status": "ok", "message": "Qdrant collection rebuilt successfully."}
