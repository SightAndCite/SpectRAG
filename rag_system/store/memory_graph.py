"""In-memory NetworkX graph store with per-session disk persistence.

A drop-in replacement for Neo4jGraphClient: the pipeline talks to its graph
store through the same duck-typed surface —

    write_graph(chunks, nx_graph)              (indexing)
    bfs_expand(seed_ids, hops, max_candidates) (query Stage 2)
    subgraph(chunk_ids)                        (query Stages 3-4)
    get_top_nodes_by_degree / get_edges_for_nodes / *_count  (viz + stats)

so nothing in IndexingPipeline / QueryPipeline / the server routes changes. The
whole graph lives in one NetworkX object that saves/loads as a single pickle, so
each chat session can own its own graph on disk (sessions/<id>/graph.pkl) and
restore it on demand — no Neo4j/Docker required.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx

_SIGNALS = ["semantic", "adjacency", "section", "entity", "utility_question", "citation"]


class InMemoryGraphStore:
    """Duck-typed stand-in for Neo4jGraphClient backed by a NetworkX graph.

    One instance holds one session's graph. Use save()/load() to persist it to a
    per-session file; the retrieval semantics (BFS, weighted subgraph) mirror the
    Neo4j client exactly, so query results are identical to the Neo4j backend.
    """

    def __init__(self) -> None:
        self._graph: nx.Graph = nx.Graph()

    # Lifecycle — no-ops so the store can be used wherever the Neo4j client is

    def connect(self) -> None:
        return None

    def close(self) -> None:
        self._graph = nx.Graph()

    @property
    def is_connected(self) -> bool:
        return True

    # Indexing — replaces the whole graph (matches Neo4j write_graph semantics)

    def write_graph(self, chunks, nx_graph: nx.Graph) -> None:
        # Copy so later mutations to the caller's graph never leak in. Node keys
        # are chunk_ids (carrying doc_id/position for the viz endpoint); edges keep
        # the `weight` + per-signal attributes GraphBuilder produced, which is
        # exactly what the query stages and the graph view expect.
        g = nx.Graph()
        for c in chunks:
            g.add_node(c.chunk_id, doc_id=c.doc_id, position=c.position)
        for u, v, attrs in nx_graph.edges(data=True):
            g.add_edge(u, v, **attrs)
        self._graph = g

    # Query Stage 2 — BFS expansion (mirrors Neo4jGraphClient.bfs_expand)

    def bfs_expand(
        self,
        seed_chunk_ids: list[str],
        hops: int,
        max_candidates: int,
    ) -> set[str]:
        visited: set[str] = set(seed_chunk_ids)
        frontier: set[str] = set(seed_chunk_ids)

        for _ in range(hops):
            if not frontier or len(visited) >= max_candidates:
                break
            new_frontier: set[str] = set()
            for node in frontier:
                if node not in self._graph:
                    continue
                for neighbor in self._graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_frontier.add(neighbor)
                        if len(visited) >= max_candidates:
                            break
                if len(visited) >= max_candidates:
                    break
            frontier = new_frontier

        return visited

    # Query Stages 3-4 — induced subgraph with weights

    def subgraph(self, chunk_ids: set[str]) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(chunk_ids)
        if not chunk_ids:
            return G
        present = [c for c in chunk_ids if c in self._graph]
        for u, v, attrs in self._graph.subgraph(present).edges(data=True):
            G.add_edge(u, v, weight=attrs.get("weight", 0.0))
        return G

    # Visualization

    def get_top_nodes_by_degree(self, limit: int) -> list[dict]:
        """Top `limit` nodes by undirected degree (mirrors the Neo4j viz query)."""
        top = sorted(self._graph.degree(), key=lambda kv: kv[1], reverse=True)[:limit]
        out: list[dict] = []
        for nid, degree in top:
            data = self._graph.nodes[nid]
            out.append({
                "id": nid,
                "doc_id": data.get("doc_id"),
                "position": data.get("position"),
                "degree": degree,
            })
        return out

    def get_edges_for_nodes(self, node_ids: set[str]) -> list[dict]:
        """All edges where both endpoints are in node_ids (each undirected edge once)."""
        if not node_ids:
            return []
        out: list[dict] = []
        for u, v, attrs in self._graph.subgraph(node_ids).edges(data=True):
            row = {"source": u, "target": v, "weight": float(attrs.get("weight", 0.0))}
            for s in _SIGNALS:
                row[s] = float(attrs.get(s, 0.0))
            out.append(row)
        return out

    # Stats

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def has_graph_data(self) -> bool:
        return self._graph.number_of_nodes() > 0

    # Persistence

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            pickle.dump(self._graph, fh, protocol=5)

    def load(self, path: Path | str) -> bool:
        """Load a graph from disk. Returns False (leaving an empty graph) if the
        file does not exist, so a fresh session degrades gracefully."""
        p = Path(path)
        if not p.exists():
            self._graph = nx.Graph()
            return False
        with open(p, "rb") as fh:
            self._graph = pickle.load(fh)
        return True
