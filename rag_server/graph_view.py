"""Serialize a session's chunk graph into the payload the 3D graph view expects.

Kept out of the route handler so the HTTP layer stays thin and this projection
logic (spectral axis selection + node/edge shaping) can be tested on its own.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from config import Config
    from rag_server.services import ActiveIndex


class GraphView:
    """Builds the /graph payload from an ActiveIndex."""

    SIGNALS = ["semantic", "adjacency", "section", "entity", "utility_question", "citation"]

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    def _spectral_axes(self, chunk_map: dict, raw_nodes: list[dict]) -> tuple[int, int, int]:
        """Pick the three highest-variance spectral dimensions for the x/y/z layout.

        sc[0..1] are unreliable when many nodes are isolated (degenerate λ≈0
        eigenvectors fill the first slots), so choose the most-varying dimensions.
        """
        sc_list = [
            chunk_map[n["id"]].spectral_coords
            for n in raw_nodes
            if chunk_map.get(n["id"]) is not None
            and chunk_map[n["id"]].spectral_coords is not None
        ]
        if len(sc_list) < 2:
            return 0, 1, 2
        sc_matrix = np.stack(sc_list).astype(np.float64)
        top3 = np.argsort(sc_matrix.var(axis=0))[-3:][::-1]
        dims = [int(d) for d in top3]
        while len(dims) < 3:
            dims.append(len(dims))
        return dims[0], dims[1], dims[2]

    def build_payload(self, active: ActiveIndex) -> dict:
        """Build the {nodes, edges, totals, has_signal_data} payload for /graph."""
        srv = self._cfg.server
        graph = active.graph
        chunk_map = {c.chunk_id: c for c in active.chunks}
        raw_nodes = graph.get_top_nodes_by_degree(srv.max_viz_nodes)
        top_set = {n["id"] for n in raw_nodes}

        dim0, dim1, dim2 = self._spectral_axes(chunk_map, raw_nodes)

        nodes = []
        for n in raw_nodes:
            chunk = chunk_map.get(n["id"])
            if not chunk:
                continue
            sc = chunk.spectral_coords
            nodes.append({
                "id":           n["id"],
                "label":        chunk.text[: srv.node_label_chars].replace("\n", " "),
                "text":         chunk.text,
                "doc":          Path(chunk.metadata.get("source", chunk.doc_id)).name,
                "position":     chunk.position,
                "degree":       n["degree"],
                "section_path": chunk.section_path,
                "sx":           float(sc[dim0]) if sc is not None and len(sc) > dim0 else 0.0,
                "sy":           float(sc[dim1]) if sc is not None and len(sc) > dim1 else 0.0,
                "sz":           float(sc[dim2]) if sc is not None and len(sc) > dim2 else 0.0,
            })

        has_signal_data = False
        edges = []
        for e in graph.get_edges_for_nodes(top_set):
            signals = {s: round(float(e.get(s) or 0.0), 4) for s in self.SIGNALS}
            if any(signals.values()):
                has_signal_data = True
            dominant = max(signals, key=signals.get) if any(signals.values()) else None
            edges.append({
                "source":   e["source"],
                "target":   e["target"],
                "weight":   round(float(e.get("weight") or 0.0), 4),
                "signals":  signals,
                "dominant": dominant,
            })

        return {
            "nodes":           nodes,
            "edges":           edges,
            "total_nodes":     graph.node_count(),
            "total_edges":     graph.edge_count(),
            "has_signal_data": has_signal_data,
        }
