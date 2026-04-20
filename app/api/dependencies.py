from __future__ import annotations

from functools import lru_cache

from app.agents.dynamic_orchestrator import DynamicOrchestrator


@lru_cache(maxsize=1)
def get_orchestrator() -> DynamicOrchestrator:
    return DynamicOrchestrator()