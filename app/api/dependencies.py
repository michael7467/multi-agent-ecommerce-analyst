from __future__ import annotations

from functools import lru_cache

from app.agents.langgraph_orchestrator import LangGraphOrchestrator


@lru_cache(maxsize=1)
def get_orchestrator() -> LangGraphOrchestrator:
    return LangGraphOrchestrator()