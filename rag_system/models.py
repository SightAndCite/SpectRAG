from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    position: int                   # 0-indexed position within document
    section_path: list[str]         # e.g. ["1. Introduction", "1.2 Background"]
    references: list[str]           # raw citation strings found in text
    language: str = "en"                 # ISO 639-1 code detected at index time
    embedding: Optional[np.ndarray] = None
    spectral_coords: Optional[np.ndarray] = None
    utility_questions: list[str] = field(default_factory=list)  # LLM "what does this answer?" (Stage 1 query-side)
    cluster_label: int = -1              # assigned offline by SpectralClusterAssigner
    metadata: dict = field(default_factory=dict)  # source path, page number, etc.

    def __hash__(self) -> int:
        return hash(self.chunk_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Chunk):
            return NotImplemented
        return self.chunk_id == other.chunk_id


@dataclass
class QueryResult:
    chunks: list[Chunk]
    answer: str
    scores: dict[str, float]        # chunk_id → final relevance score
