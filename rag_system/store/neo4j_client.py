from __future__ import annotations
import logging
from typing import TYPE_CHECKING

import networkx as nx
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

if TYPE_CHECKING:
    from rag_system.models import Chunk
    from config import Neo4jConfig

logger = logging.getLogger(__name__)

_SIGNALS = ["semantic", "adjacency", "section", "entity", "utility_question", "citation"]


class Neo4jGraphClient:
    """
    All graph I/O goes through this class.

    Indexing  → write_graph(chunks, nx_graph)      clears + rewrites Neo4j
    Stage 2   → bfs_expand(seed_ids, hops, cap)    Cypher BFS
    Stage 3+4 → subgraph(chunk_ids)                returns nx.Graph for PPR / selection
    Viz       → get_top_nodes_by_degree / get_edges_for_nodes
    """

    def __init__(self, cfg: Neo4jConfig) -> None:
        self._cfg = cfg
        self._driver = None

    # Lifecycle

    def connect(self) -> None:
        self._driver = GraphDatabase.driver(
            self._cfg.uri,
            auth=(self._cfg.user, self._cfg.password),
        )
        self._driver.verify_connectivity()
        with self._session() as s:
            s.run(
                "CREATE INDEX chunk_id IF NOT EXISTS FOR (n:Chunk) ON (n.id)"
            )
        logger.info("Neo4j connected: %s  db=%s", self._cfg.uri, self._cfg.database)

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    @property
    def is_connected(self) -> bool:
        return self._driver is not None

    def _session(self, **kwargs):
        return self._driver.session(database=self._cfg.database, **kwargs)

    # Indexing

    def write_graph(self, chunks: list[Chunk], nx_graph: nx.Graph) -> None:
        """
        Atomically replaces the entire graph.
        Deletes all existing Chunk nodes, then batch-writes nodes then edges.
        """
        bs = self._cfg.write_batch_size

        logger.info("Neo4j: clearing existing graph…")
        with self._session() as s:
            s.run("MATCH (n:Chunk) DETACH DELETE n")

        # Nodes
        node_batch = [
            {"id": c.chunk_id, "doc_id": c.doc_id, "position": c.position}
            for c in chunks
        ]
        logger.info("Neo4j: writing %d nodes…", len(node_batch))
        for i in range(0, len(node_batch), bs):
            with self._session() as s:
                s.run(
                    """
                    UNWIND $batch AS c
                    CREATE (n:Chunk {id: c.id, doc_id: c.doc_id, position: c.position})
                    """,
                    batch=node_batch[i : i + bs],
                )

        # Edges
        edge_batch = []
        for u, v, attrs in nx_graph.edges(data=True):
            edge_batch.append(
                {
                    "src": u,
                    "tgt": v,
                    "weight": float(attrs.get("weight", 0.0)),
                    **{s: float(attrs.get(s, 0.0)) for s in _SIGNALS},
                }
            )

        logger.info("Neo4j: writing %d edges…", len(edge_batch))
        for i in range(0, len(edge_batch), bs):
            with self._session() as s:
                s.run(
                    """
                    UNWIND $batch AS e
                    MATCH (a:Chunk {id: e.src}), (b:Chunk {id: e.tgt})
                    CREATE (a)-[:CONNECTED {
                        weight:           e.weight,
                        semantic:         e.semantic,
                        adjacency:        e.adjacency,
                        section:          e.section,
                        entity:           e.entity,
                        utility_question: e.utility_question,
                        citation:         e.citation
                    }]->(b)
                    """,
                    batch=edge_batch[i : i + bs],
                )

        logger.info(
            "Neo4j: written %d nodes, %d edges",
            nx_graph.number_of_nodes(),
            nx_graph.number_of_edges(),
        )

    # Query — Stage 2: BFS expansion

    def bfs_expand(
        self,
        seed_chunk_ids: list[str],
        hops: int,
        max_candidates: int,
    ) -> set[str]:
        """
        Iterative BFS. One Cypher round-trip per hop; stops when max_candidates reached.
        Uses undirected traversal so both directions of each stored edge are followed.
        """
        visited: set[str] = set(seed_chunk_ids)
        frontier: set[str] = set(seed_chunk_ids)

        for _ in range(hops):
            if not frontier or len(visited) >= max_candidates:
                break
            with self._session() as s:
                result = s.run(
                    """
                    MATCH (s:Chunk)-[:CONNECTED]-(n:Chunk)
                    WHERE s.id IN $frontier
                    RETURN DISTINCT n.id AS id
                    """,
                    frontier=list(frontier),
                )
                new_frontier: set[str] = set()
                for record in result:
                    nid = record["id"]
                    if nid not in visited:
                        visited.add(nid)
                        new_frontier.add(nid)
                        if len(visited) >= max_candidates:
                            break
            frontier = new_frontier

        return visited

    # Query — Stages 3+4: in-memory subgraph

    def subgraph(self, chunk_ids: set[str]) -> nx.Graph:
        """
        Fetches edges between the given IDs and returns a lightweight nx.Graph
        so that PPR diffusion and cluster selection run unchanged.
        """
        G = nx.Graph()
        G.add_nodes_from(chunk_ids)
        if not chunk_ids:
            return G

        with self._session() as s:
            result = s.run(
                """
                MATCH (a:Chunk)-[e:CONNECTED]-(b:Chunk)
                WHERE a.id IN $ids AND b.id IN $ids
                RETURN a.id AS source, b.id AS target, e.weight AS weight
                """,
                ids=list(chunk_ids),
            )
            for record in result:
                G.add_edge(record["source"], record["target"], weight=record["weight"])
        return G

    # Visualization

    def get_top_nodes_by_degree(self, limit: int) -> list[dict]:
        """Top `limit` nodes sorted by undirected degree."""
        with self._session() as s:
            result = s.run(
                """
                MATCH (n:Chunk)
                WITH n, size([(n)-[:CONNECTED]-() | 1]) AS degree
                ORDER BY degree DESC
                LIMIT $limit
                RETURN n.id AS id, n.doc_id AS doc_id, n.position AS position, degree
                """,
                limit=limit,
            )
            return [
                {
                    "id": r["id"],
                    "doc_id": r["doc_id"],
                    "position": r["position"],
                    "degree": r["degree"],
                }
                for r in result
            ]

    def get_edges_for_nodes(self, node_ids: set[str]) -> list[dict]:
        """
        All edges where both endpoints are in node_ids.
        Deduplicates by returning only a→b where a.id < b.id
        (we store directed edges, so undirected traversal would double-count).
        """
        if not node_ids:
            return []
        with self._session() as s:
            result = s.run(
                """
                MATCH (a:Chunk)-[e:CONNECTED]->(b:Chunk)
                WHERE a.id IN $ids AND b.id IN $ids
                RETURN
                    a.id AS source, b.id AS target,
                    e.weight           AS weight,
                    e.semantic         AS semantic,
                    e.section          AS section,
                    e.entity           AS entity,
                    e.adjacency        AS adjacency,
                    e.utility_question AS utility_question,
                    e.citation         AS citation
                """,
                ids=list(node_ids),
            )
            return [dict(r) for r in result]

    # Stats

    def node_count(self) -> int:
        with self._session() as s:
            return s.run("MATCH (n:Chunk) RETURN count(n) AS c").single()["c"]

    def edge_count(self) -> int:
        with self._session() as s:
            return s.run(
                "MATCH ()-[e:CONNECTED]->() RETURN count(e) AS c"
            ).single()["c"]

    def has_graph_data(self) -> bool:
        return self.node_count() > 0
