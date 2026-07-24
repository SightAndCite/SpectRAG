from __future__ import annotations
from rag_system.models import Chunk
from rag_system.indexing.edges.base import EdgeExtractor, RawEdge


class AdjacencyEdgeExtractor(EdgeExtractor):
    """Full weight (1.0) edge between consecutive chunks in the same document."""

    def extract(self, chunks: list[Chunk]) -> list[RawEdge]:
        edges: list[RawEdge] = []
        for i in range(len(chunks) - 1):
            ci, cj = chunks[i], chunks[i + 1]
            if ci.doc_id == cj.doc_id and cj.position == ci.position + 1:
                edges.append(RawEdge(i, i + 1, 1.0))
        return edges
