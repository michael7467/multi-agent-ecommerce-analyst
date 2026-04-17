from __future__ import annotations

import os

import redis


_REDIS_CLIENT: redis.Redis | None = None


def get_redis_client() -> redis.Redis:
    global _REDIS_CLIENT

    if _REDIS_CLIENT is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _REDIS_CLIENT = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
        )

    return _REDIS_CLIENT