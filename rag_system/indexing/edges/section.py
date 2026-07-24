from __future__ import annotations
from collections import defaultdict
from rag_system.models import Chunk
from rag_system.indexing.edges.base import EdgeExtractor, RawEdge
from config import IndexingConfig


class SectionEdgeExtractor(EdgeExtractor):
    """Edges between chunks sharing a section-path prefix. Deeper shared = stronger."""

    def __init__(self, cfg: IndexingConfig) -> None:
        self.max_members = cfg.max_section_members  # cap to avoid O(N²) in large sections

    def extract(self, chunks: list[Chunk]) -> list[RawEdge]:
        # Group chunk indices by each prefix of their section_path
        depth_groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for i, chunk in enumerate(chunks):
            path = tuple(chunk.section_path)
            for depth in range(1, len(path) + 1):
                depth_groups[path[:depth]].append(i)

        edges: dict[tuple[int, int], float] = {}
        for prefix, members in depth_groups.items():
            if len(members) < 2:
                continue
            depth = len(prefix)
            # Cap members to prevent quadratic blowup in monolithic sections
            capped = members[: self.max_members]
            for a in range(len(capped)):
                for b in range(a + 1, len(capped)):
                    i, j = capped[a], capped[b]
                    max_depth = max(
                        len(chunks[i].section_path),
                        len(chunks[j].section_path),
                        1,
                    )
                    score = depth / max_depth
                    key = (min(i, j), max(i, j))
                    if score > edges.get(key, -1.0):
                        edges[key] = score

        return [RawEdge(i, j, s) for (i, j), s in edges.items()]
