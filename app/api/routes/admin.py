from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.middleware.auth_middleware import require_admin
from app.rag.qdrant_index_builder import QdrantIndexBuilder
from app.logging.logger import get_logger

logger = get_logger("api.admin")

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/rebuild-index", dependencies=[Depends(require_admin)])
def rebuild_index() -> dict:
   
    builder = QdrantIndexBuilder()

    try:
        builder.build()
    except Exception as exc:
        logger.error("Index rebuild failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Index rebuild failed: {exc}") from exc

    return {"status": "ok", "message": "Qdrant collection rebuilt successfully."}
