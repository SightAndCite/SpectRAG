from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from rag_system.models import Chunk
from rag_system.indexing.chunker import DocumentChunker
from rag_system.indexing.embedder import OllamaEmbedder
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

    #: Every signal the graph builder understands, in its canonical order.
    SIGNALS = ("semantic", "adjacency", "section", "entity", "utility_question", "citation")

    def __init__(self, cfg: Config, neo4j_client: Neo4jGraphClient) -> None:
        self.cfg = cfg
        self.neo4j_client = neo4j_client
        self.embedder = OllamaEmbedder(cfg.ollama)
        self.chunker = DocumentChunker(cfg.indexing, cfg.language, embedder=self.embedder)
        self.graph_builder = GraphBuilder(cfg)

        unknown = set(cfg.indexing.enabled_edge_signals) - set(self.SIGNALS)
        if unknown:
            raise ValueError(
                f"Unknown edge signal(s) in enabled_edge_signals: {sorted(unknown)}. "
                f"Valid values: {list(self.SIGNALS)}"
            )
        self.enabled_signals = tuple(
            s for s in self.SIGNALS if s in set(cfg.indexing.enabled_edge_signals)
        )

        # Disabled stages are left as None rather than constructed-and-skipped, so
        # calling one raises instead of silently doing work, and so their clients
        # (and credentials) are never created.
        self.spectral = (
            SpectralComputer(cfg.indexing) if cfg.indexing.spectral_enabled else None
        )
        self.cluster_assigner = (
            SpectralClusterAssigner(cfg.indexing)
            if cfg.indexing.clustering_enabled else None
        )

    def _extract_signals(self, chunks: list[Chunk], report: ProgressCb) -> dict:
        """Run only the enabled edge extractors.

        Imports are local so a disabled signal never loads its module either: a
        dense-only profile should not require spaCy, PyTorch or an API key just
        because some other extractor exists in the tree.
        """
        cfg = self.cfg
        out: dict[str, list] = {name: [] for name in self.SIGNALS}
        skipped = [s for s in self.SIGNALS if s not in self.enabled_signals]
        if skipped:
            logger.info("Edge signals disabled (not constructed): %s", ", ".join(skipped))

        if "semantic" in self.enabled_signals:
            report("Edges: semantic")
            from rag_system.indexing.edges.semantic import SemanticEdgeExtractor
            out["semantic"] = SemanticEdgeExtractor(cfg.indexing).extract(chunks)
        if "adjacency" in self.enabled_signals:
            report("Edges: adjacency")
            from rag_system.indexing.edges.adjacency import AdjacencyEdgeExtractor
            out["adjacency"] = AdjacencyEdgeExtractor().extract(chunks)
        if "section" in self.enabled_signals:
            report("Edges: section")
            from rag_system.indexing.edges.section import SectionEdgeExtractor
            out["section"] = SectionEdgeExtractor(cfg.indexing).extract(chunks)
        if "entity" in self.enabled_signals:
            report(f"Edges: entity (NER on {len(chunks)} chunks)")
            from rag_system.indexing.edges.entity import EntityEdgeExtractor
            out["entity"] = EntityEdgeExtractor(
                cfg.indexing, cfg.ner, cfg.language, cfg.ollama, cfg.openai
            ).extract(chunks)
        if "utility_question" in self.enabled_signals:
            report(f"Edges: utility-questions ({len(chunks)} LLM calls)")
            from rag_system.indexing.edges.utility_question import UtilityQuestionEdgeExtractor
            out["utility_question"] = UtilityQuestionEdgeExtractor(
                cfg.indexing, cfg.ollama, self.embedder, cfg.openai
            ).extract(chunks)
        if "citation" in self.enabled_signals:
            report("Edges: citation")
            from rag_system.indexing.edges.citation import CitationEdgeExtractor
            out["citation"] = CitationEdgeExtractor(cfg.indexing).extract(chunks)
        return out

    def index(
        self,
        paths: list[Path | str],
        progress_cb: ProgressCb | None = None,
        doc_root: Path | str | None = None,
        frozen_transforms=None,
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
            all_chunks.extend(self.chunker.chunk_file(path, doc_root=doc_root))

        if not all_chunks:
            raise ValueError("No chunks produced — check file paths and formats.")
        _report(f"Chunking done — {len(all_chunks)} chunks")

        # 2. Embed
        _report(f"Embedding {len(all_chunks)} chunks…")
        self.embedder.embed_chunks(all_chunks)
        ic = self.cfg.indexing
        faiss_index = self.embedder.build_faiss_index(
            all_chunks, index_type=ic.vector_index_type, m=ic.hnsw_m,
            ef_construction=ic.hnsw_ef_construction, ef_search=ic.hnsw_ef_search)
        _report("Embedding done")

        # 3. Edge signals — only the enabled ones are constructed or run.
        signal_map = self._extract_signals(all_chunks, _report)
        semantic = signal_map["semantic"]
        adjacency = signal_map["adjacency"]
        section = signal_map["section"]
        entity = signal_map["entity"]
        uq = signal_map["utility_question"]
        citation = signal_map["citation"]

        logger.info(
            "Edges — sem:%d adj:%d sec:%d ent:%d uq:%d cite:%d",
            len(semantic), len(adjacency), len(section),
            len(entity), len(uq), len(citation),
        )

        # 4. Build graph (NetworkX, for spectral decomp)
        # Choose per-signal edge weights by mode: hardcoded priors ("fixed"),
        # informativeness heuristic ("calibrate", Stage A), or self-supervised
        # logistic regression ("fit", Stage B).
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
        # A delta passes the base generation's transforms so its edges land on
        # the same scale; a full build passes None and fits fresh ones, which
        # `self.graph_builder.transforms` then reports for the manifest.
        graph = self.graph_builder.build(
            all_chunks, semantic, adjacency, section, entity, uq, citation,
            weights=edge_weights, frozen=frozen_transforms,
        )

        # 5. Spectral decomposition
        # "multiplex" (Option 4) derives coords from per-signal Laplacians scaled
        # by the chosen weights; "combined" uses the single combined-weight graph.
        if self.spectral is None:
            _report("Spectral coordinates: disabled — Stage 3 will use PPR only")
        else:
            _report("Computing spectral coordinates…")
            if self.cfg.indexing.spectral_mode == "multiplex":
                self.spectral.compute_multiplex(signal_map, edge_weights, all_chunks)
            else:
                self.spectral.compute(graph, all_chunks)

        # 5b. Assign stable global cluster labels once (reused at query time for
        # the cluster-novelty bonus, so Stage 4 need not re-run K-Means per query).
        if self.cluster_assigner is None:
            _report("Cluster labels: disabled")
        else:
            _report("Assigning spectral cluster labels…")
            self.cluster_assigner.assign(all_chunks)

        # 6. Write graph to Neo4j
        _report("Writing graph to Neo4j…")
        self.neo4j_client.write_graph(all_chunks, graph)

        # 7. Persist chunks + FAISS to disk
        _report("Saving index…")
        store = IndexStore(self.cfg.store_path, self.cfg.indexing)
        store.save(all_chunks, faiss_index, embedder=self.embedder)

        _report(
            f"Done — {len(all_chunks)} chunks, "
            f"{graph.number_of_edges()} edges in Neo4j"
        )
        return store
