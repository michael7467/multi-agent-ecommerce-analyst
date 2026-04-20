from __future__ import annotations

import hashlib
import json
from typing import Any

from app.cache.redis_client import get_redis_client
from app.logging.logger import get_logger
from app.observability.tracing import get_tracer

logger = get_logger("services.cache")


class CacheService:
    def __init__(self) -> None:
        self.client = get_redis_client()
        self.tracer = get_tracer("app.cache_service")

    def _make_key(self, prefix: str, payload: dict[str, Any]) -> str:
        if not isinstance(prefix, str) or not prefix.strip():
            raise ValueError("prefix must be a non-empty string")

        if not isinstance(payload, dict):
            raise ValueError("payload must be a dictionary")

        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"{prefix}:{digest}"

    def get_json(self, prefix: str, payload: dict[str, Any]) -> Any | None:
        key = self._make_key(prefix, payload)

        with self.tracer.start_as_current_span("cache.get"):
            try:
                cached = self.client.get(key)
            except Exception:
                logger.error(f"Redis GET failed for key={key}", exc_info=True)
                return None

            if cached is None:
                logger.debug(f"Cache miss: {key}")
                return None

            logger.debug(f"Cache hit: {key}")
            return json.loads(cached)

    def set_json(
        self,
        prefix: str,
        payload: dict[str, Any],
        value: Any,
        ttl_seconds: int = 1800,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a positive integer")

        key = self._make_key(prefix, payload)
        serialized = json.dumps(value, ensure_ascii=False)

        with self.tracer.start_as_current_span("cache.set"):
            try:
                self.client.set(key, serialized, ex=ttl_seconds)
                logger.debug(f"Cache set: {key} (ttl={ttl_seconds})")
            except Exception:
                logger.error(f"Redis SET failed for key={key}", exc_info=True)

    def delete(self, prefix: str, payload: dict[str, Any]) -> None:
        key = self._make_key(prefix, payload)

        with self.tracer.start_as_current_span("cache.delete"):
            try:
                self.client.delete(key)
                logger.debug(f"Cache delete: {key}")
            except Exception:
                logger.error(f"Redis DELETE failed for key={key}", exc_info=True)
