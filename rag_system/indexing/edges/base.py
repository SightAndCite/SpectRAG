from __future__ import annotations
from abc import ABC, abstractmethod
from typing import NamedTuple
from rag_system.models import Chunk


class RawEdge(NamedTuple):
    i: int          # index into the global chunks list
    j: int
    score: float    # raw signal strength in [0, 1]


class EdgeExtractor(ABC):
    @abstractmethod
    def extract(self, chunks: list[Chunk]) -> list[RawEdge]:
        """Return raw edges for this signal. i < j is not required."""
        ...
