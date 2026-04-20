from __future__ import annotations

import os

import redis
from app.core.config import settings

_REDIS_CLIENT: redis.Redis | None = None


def get_redis_client() -> redis.Redis:
    global _REDIS_CLIENT

    if _REDIS_CLIENT is None:
        redis_url = settings.redis_url
        _REDIS_CLIENT = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
        )

    return _REDIS_CLIENT