from __future__ import annotations
import logging
import networkx as nx
from rag_system.models import Chunk
from rag_system.indexing.edges.base import RawEdge
from rag_system.indexing.transforms import SignalTransform, TransformSet
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
        #: Transforms actually used by the last build, for the generation
        #: manifest. Recomputing extrema on every build rescales every edge
        #: already published when a stronger one arrives, which silently
        #: rewrites the base graph's topology on an incremental update.
        self.transforms = TransformSet()

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
        frozen: TransformSet | None = None,
    ) -> nx.Graph:
        """Combine the signals into one weighted graph.

        `frozen` carries the transforms a previous generation was built with. A
        delta reuses them so its edges land on the base's scale; a full build
        passes None, fits fresh ones, and records them for the next delta.
        """
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

        # Copy so excursion counters are per build; `apply` mutates them.
        carried = frozen.fresh() if frozen else None
        used = TransformSet()
        for name, raw_edges, weight in signal_configs:
            transform = (carried.signals.get(name) if carried else None)
            if transform is None:
                transform = SignalTransform.fit(e.score for e in raw_edges)
            used.signals[name] = transform
            norm_map = _normalize_with(raw_edges, transform)
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

        self.transforms = used
        excursions = used.excursions()
        if excursions:
            # Clipped rather than widening the range, because widening is the
            # rescale being avoided. Visible so drift can justify a rebuild.
            logger.warning(
                "Delta scores fell outside the frozen transform range and were "
                "clipped: %s. Base edge weights are unchanged; a full rebuild "
                "refits the range.", excursions)
        logger.info(
            "Graph: %d nodes, %d edges (dropped %d below threshold %.3f)",
            G.number_of_nodes(), kept, len(combined) - kept, self.threshold,
        )
        return G


def _normalize_with(edges: list[RawEdge],
                    transform: SignalTransform) -> dict[tuple[int, int], float]:
    """Apply a fixed transform, keeping the strongest score per undirected pair."""
    result: dict[tuple[int, int], float] = {}
    for e in edges:
        key = (min(e.i, e.j), max(e.i, e.j))
        norm = transform.apply(e.score)
        if norm > result.get(key, -1.0):
            result[key] = norm
    return result


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
