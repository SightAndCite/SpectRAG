from __future__ import annotations
import numpy as np
import faiss
from rag_system.models import Chunk
from rag_system.indexing.edges.base import EdgeExtractor, RawEdge
from config import IndexingConfig


class SemanticEdgeExtractor(EdgeExtractor):
    """k-NN cosine-similarity edges kept only within a neighborhood (sparse)."""

    def __init__(self, cfg: IndexingConfig) -> None:
        self.k = cfg.semantic_k_neighbors
        self.threshold = cfg.semantic_threshold

    def extract(self, chunks: list[Chunk]) -> list[RawEdge]:
        if len(chunks) < 2:
            return []

        matrix = np.stack([c.embedding for c in chunks]).astype(np.float32)
        dim = matrix.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)

        k = min(self.k + 1, len(chunks))   # +1 because each chunk is its own NN
        scores_mat, idx_mat = index.search(matrix, k)

        edges: dict[tuple[int, int], float] = {}
        for i, (nbrs, sims) in enumerate(zip(idx_mat, scores_mat)):
            for j, sim in zip(nbrs, sims):
                if j == i or sim < self.threshold:
                    continue
                key = (min(i, j), max(i, j))
                if sim > edges.get(key, -1.0):
                    edges[key] = float(sim)

        return [RawEdge(i, j, s) for (i, j), s in edges.items()]
