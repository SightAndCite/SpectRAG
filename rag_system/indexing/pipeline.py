from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from rag_system.models import Chunk
from rag_system.indexing.chunker import DocumentChunker
from rag_system.indexing.embedder import OllamaEmbedder
from rag_system.indexing.edges.semantic import SemanticEdgeExtractor
from rag_system.indexing.edges.adjacency import AdjacencyEdgeExtractor
from rag_system.indexing.edges.section import SectionEdgeExtractor
from rag_system.indexing.edges.entity import EntityEdgeExtractor
from rag_system.indexing.edges.utility_question import UtilityQuestionEdgeExtractor
from rag_system.indexing.edges.citation import CitationEdgeExtractor
from rag_system.indexing.graph_builder import GraphBuilder
from rag_system.indexing.weight_calibrator import calibrate_edge_weights, fit_edge_weights
from rag_system.indexing.spectral import SpectralComputer
from rag_system.indexing.cluster_assigner import SpectralClusterAssigner
from rag_system.store.index_store import IndexStore
from config import Config

if TYPE_CHECKING:
    from rag_system.store.neo4j_client import Neo4jGraphClient

logger = logging.getLogger(__name__)

ProgressCb = Callable[[str], None]


class IndexingPipeline:
    """
    Orchestrates the full offline indexing pass:
      chunk → embed → six edge signals → graph → spectral → persist

    The NetworkX graph built here is used only for spectral decomposition;
    the persistent graph store is Neo4j (via neo4j_client.write_graph).
    """

    def __init__(self, cfg: Config, neo4j_client: Neo4jGraphClient) -> None:
        self.cfg = cfg
        self.neo4j_client = neo4j_client
        self.embedder = OllamaEmbedder(cfg.ollama)
        self.chunker = DocumentChunker(cfg.indexing, cfg.language, embedder=self.embedder)
        self.graph_builder = GraphBuilder(cfg)
        self.spectral = SpectralComputer(cfg.indexing)
        self.cluster_assigner = SpectralClusterAssigner(cfg.indexing)

    def index(
        self,
        paths: list[Path | str],
        progress_cb: ProgressCb | None = None,
    ) -> IndexStore:
        def _report(msg: str) -> None:
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        # 1. Chunk
        all_chunks: list[Chunk] = []
        for p in paths:
            path = Path(p)
            if not path.exists():
                logger.error("File not found, skipping: %s", path)
                continue
            _report(f"Chunking: {path.name}")
            all_chunks.extend(self.chunker.chunk_file(path))

        if not all_chunks:
            raise ValueError("No chunks produced — check file paths and formats.")
        _report(f"Chunking done — {len(all_chunks)} chunks")

        # 2. Embed
        _report(f"Embedding {len(all_chunks)} chunks…")
        self.embedder.embed_chunks(all_chunks)
        faiss_index = self.embedder.build_faiss_index(all_chunks)
        _report("Embedding done")

        # 3. Edge signals
        _report("Edges: semantic")
        semantic = SemanticEdgeExtractor(self.cfg.indexing).extract(all_chunks)

        _report("Edges: adjacency + section")
        adjacency = AdjacencyEdgeExtractor().extract(all_chunks)
        section = SectionEdgeExtractor(self.cfg.indexing).extract(all_chunks)

        _report(f"Edges: entity (NER on {len(all_chunks)} chunks)")
        entity = EntityEdgeExtractor(
            self.cfg.indexing, self.cfg.ner, self.cfg.language, self.cfg.ollama, self.cfg.openai
        ).extract(all_chunks)

        _report(f"Edges: utility-questions ({len(all_chunks)} LLM calls)")
        uq = UtilityQuestionEdgeExtractor(
            self.cfg.indexing, self.cfg.ollama, self.embedder, self.cfg.openai
        ).extract(all_chunks)

        _report("Edges: citation")
        citation = CitationEdgeExtractor().extract(all_chunks)

        logger.info(
            "Edges — sem:%d adj:%d sec:%d ent:%d uq:%d cite:%d",
            len(semantic), len(adjacency), len(section),
            len(entity), len(uq), len(citation),
        )

        # 4. Build graph (NetworkX, for spectral decomp)
        # Choose per-signal edge weights by mode: hardcoded priors ("fixed"),
        # informativeness heuristic ("calibrate", Stage A), or self-supervised
        # logistic regression ("fit", Stage B).
        signal_map = {
            "semantic": semantic,
            "adjacency": adjacency,
            "section": section,
            "entity": entity,
            "utility_question": uq,
            "citation": citation,
        }
        mode = self.cfg.indexing.edge_weight_mode
        if mode == "calibrate":
            _report("Calibrating edge weights (Stage A)…")
            edge_weights = calibrate_edge_weights(signal_map, self.cfg.edge_weights)
        elif mode == "fit":
            _report("Fitting edge weights (Stage B)…")
            edge_weights = fit_edge_weights(
                signal_map, self.cfg.edge_weights, n_chunks=len(all_chunks)
            )
        else:
            edge_weights = self.cfg.edge_weights

        _report("Building graph…")
        graph = self.graph_builder.build(
            all_chunks, semantic, adjacency, section, entity, uq, citation,
            weights=edge_weights,
        )

        # 5. Spectral decomposition
        # "multiplex" (Option 4) derives coords from per-signal Laplacians scaled
        # by the chosen weights; "combined" uses the single combined-weight graph.
        _report("Computing spectral coordinates…")
        if self.cfg.indexing.spectral_mode == "multiplex":
            self.spectral.compute_multiplex(signal_map, edge_weights, all_chunks)
        else:
            self.spectral.compute(graph, all_chunks)

        # 5b. Assign stable global cluster labels once (reused at query time for
        # the cluster-novelty bonus, so Stage 4 need not re-run K-Means per query).
        _report("Assigning spectral cluster labels…")
        self.cluster_assigner.assign(all_chunks)

        # 6. Write graph to Neo4j
        _report("Writing graph to Neo4j…")
        self.neo4j_client.write_graph(all_chunks, graph)

        # 7. Persist chunks + FAISS to disk
        _report("Saving index…")
        store = IndexStore(self.cfg.store_path)
        store.save(all_chunks, faiss_index)

        _report(
            f"Done — {len(all_chunks)} chunks, "
            f"{graph.number_of_edges()} edges in Neo4j"
        )
        return store
