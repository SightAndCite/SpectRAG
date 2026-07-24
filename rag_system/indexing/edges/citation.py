from __future__ import annotations
import re
from collections import defaultdict
from rag_system.models import Chunk
from rag_system.indexing.edges.base import EdgeExtractor, RawEdge

_CITE_BRACKET_RE = re.compile(r"\[(\d+(?:[,;]\s*\d+)*)\]")
_DOI_RE = re.compile(r"doi:\s*(10\.\d{4,}/\S+)", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s>\"']+")


class CitationEdgeExtractor(EdgeExtractor):
    """
    Edges between chunks that share citation keys (bracket refs, DOIs, URLs).
    Directional signal (one cites the other) is symmetrized in the graph builder.
    """

    def extract(self, chunks: list[Chunk]) -> list[RawEdge]:
        key_to_chunks: dict[str, list[int]] = defaultdict(list)
        for i, chunk in enumerate(chunks):
            for key in self._citation_keys(chunk.text):
                key_to_chunks[key].append(i)

        edge_counts: dict[tuple[int, int], float] = {}
        for key, chunk_idxs in key_to_chunks.items():
            if len(chunk_idxs) < 2:
                continue
            for a in range(len(chunk_idxs)):
                for b in range(a + 1, len(chunk_idxs)):
                    pair = (chunk_idxs[a], chunk_idxs[b])
                    edge_counts[pair] = edge_counts.get(pair, 0.0) + 1.0

        if not edge_counts:
            return []

        max_count = max(edge_counts.values())
        return [RawEdge(i, j, count / max_count) for (i, j), count in edge_counts.items()]

    @staticmethod
    def _citation_keys(text: str) -> set[str]:
        keys: set[str] = set()
        for m in _CITE_BRACKET_RE.finditer(text):
            for num in re.split(r"[,;]\s*", m.group(1)):
                keys.add(f"ref:{num.strip()}")
        for m in _DOI_RE.finditer(text):
            keys.add(f"doi:{m.group(1).lower()}")
        for url in _URL_RE.findall(text):
            keys.add(f"url:{url.rstrip('.,;)')}")
        return keys
