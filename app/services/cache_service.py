from __future__ import annotations

import hashlib
import json
from typing import Any

from app.cache.redis_client import get_redis_client


class CacheService:
    def __init__(self) -> None:
        self.client = get_redis_client()

    def _make_key(self, prefix: str, payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"{prefix}:{digest}"

    def get_json(self, prefix: str, payload: dict[str, Any]) -> Any | None:
        key = self._make_key(prefix, payload)
        cached = self.client.get(key)
        if cached is None:
            return None
        return json.loads(cached)

    def set_json(
        self,
        prefix: str,
        payload: dict[str, Any],
        value: Any,
        ttl_seconds: int = 1800,
    ) -> None:
        key = self._make_key(prefix, payload)
        serialized = json.dumps(value, ensure_ascii=False)
        self.client.set(key, serialized, ex=ttl_seconds)

    def delete(self, prefix: str, payload: dict[str, Any]) -> None:
        key = self._make_key(prefix, payload)
        self.client.delete(key)