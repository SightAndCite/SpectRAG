from __future__ import annotations
from typing import TYPE_CHECKING

from rag_system.models import Chunk
from config import Config

if TYPE_CHECKING:
    from rag_system.store.neo4j_client import Neo4jGraphClient


class GraphExpander:
    """
    Stage 2 — BFS from seeds to collect multi-hop neighbors.
    Delegates traversal to Neo4j via Cypher so the full graph never needs to
    be loaded into Python memory.
    """

    def __init__(self, cfg: Config) -> None:
        self.hops = cfg.retrieval.graph_expansion_hops
        self.max_candidates = cfg.retrieval.max_candidates

    def expand(
        self,
        seed_indices: list[int],
        chunks: list[Chunk],
        neo4j_client: Neo4jGraphClient,
    ) -> list[int]:
        chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(chunks)}
        seed_chunk_ids = [chunks[i].chunk_id for i in seed_indices]

        expanded_ids = neo4j_client.bfs_expand(
            seed_chunk_ids, self.hops, self.max_candidates
        )

        return [
            chunk_id_to_idx[cid]
            for cid in expanded_ids
            if cid in chunk_id_to_idx
        ]
