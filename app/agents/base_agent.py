from __future__ import annotations

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError