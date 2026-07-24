from __future__ import annotations
import logging
import networkx as nx
from rag_system.models import Chunk
from rag_system.indexing.edges.base import RawEdge
from config import Config, EdgeWeights

logger = logging.getLogger(__name__)

_SIGNALS = ["semantic", "adjacency", "section", "entity", "utility_question", "citation"]


class GraphBuilder:
    """
    Combines six edge signals into a single weighted undirected graph.
    Per-signal normalized scores are stored as edge attributes for visualization.
    """

    def __init__(self, cfg: Config) -> None:
        self.weights = cfg.edge_weights
        self.threshold = cfg.indexing.edge_sparsify_threshold

    def build(
        self,
        chunks: list[Chunk],
        semantic: list[RawEdge],
        adjacency: list[RawEdge],
        section: list[RawEdge],
        entity: list[RawEdge],
        utility_question: list[RawEdge],
        citation: list[RawEdge],
        weights: EdgeWeights | None = None,
    ) -> nx.Graph:
        w = weights or self.weights
        signal_configs = [
            ("semantic",         semantic,         w.semantic),
            ("adjacency",        adjacency,        w.adjacency),
            ("section",          section,          w.section),
            ("entity",           entity,           w.entity),
            ("utility_question", utility_question, w.utility_question),
            ("citation",         citation,         w.citation),
        ]

        combined: dict[tuple[int, int], dict] = {}

        for name, raw_edges, weight in signal_configs:
            norm_map = _minmax_normalize(raw_edges)
            for (i, j), score in norm_map.items():
                if (i, j) not in combined:
                    combined[(i, j)] = {"weight": 0.0}
                combined[(i, j)]["weight"] += weight * score
                combined[(i, j)][name] = round(score, 4)   # store for visualization

        chunk_ids = [c.chunk_id for c in chunks]
        G = nx.Graph()
        G.add_nodes_from(chunk_ids)

        kept = 0
        for (i, j), data in combined.items():
            if data["weight"] >= self.threshold:
                G.add_edge(
                    chunk_ids[i], chunk_ids[j],
                    weight=round(data["weight"], 4),
                    **{s: data.get(s, 0.0) for s in _SIGNALS},
                )
                kept += 1

        logger.info(
            "Graph: %d nodes, %d edges (dropped %d below threshold %.3f)",
            G.number_of_nodes(), kept, len(combined) - kept, self.threshold,
        )
        return G


def _minmax_normalize(edges: list[RawEdge]) -> dict[tuple[int, int], float]:
    if not edges:
        return {}
    scores = [e.score for e in edges]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return {(min(e.i, e.j), max(e.i, e.j)): 1.0 for e in edges}
    span = hi - lo
    result: dict[tuple[int, int], float] = {}
    for e in edges:
        key = (min(e.i, e.j), max(e.i, e.j))
        norm = (e.score - lo) / span
        if norm > result.get(key, -1.0):
            result[key] = norm
    return result
