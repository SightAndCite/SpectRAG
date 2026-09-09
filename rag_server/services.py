"""Server services: the active-session index cache and the indexing worker."""
from __future__ import annotations

import dataclasses
import logging
import time
import threading
from typing import TYPE_CHECKING

from rag_system.indexing.pipeline import IndexingPipeline
from rag_system.query.pipeline import QueryPipeline
from rag_system.store.generation import Manifest
from rag_system.store.index_store import IndexStore
from rag_system.store.memory_graph import InMemoryGraphStore
from rag_server.runtime import ServerPaths

if TYPE_CHECKING:
    import faiss
    from config import Config
    from rag_system.models import Chunk, QueryResult
    from rag_server.session_store import SessionStore

logger = logging.getLogger(__name__)


class ActiveIndex:
    """Holds the currently-loaded session's index + a shared query pipeline.

    Only one session's index is kept in memory at a time (this is a single-user
    local app). ``ensure_loaded`` swaps to a session on demand, restoring its
    chunks, FAISS index, and graph from disk.
    """

    def __init__(self, cfg: Config, paths: ServerPaths) -> None:
        self._cfg = cfg
        self._paths = paths
        # Keyed by CORPUS GENERATION, not by session: two chats over the same
        # documents share one loaded index, and switching between them reloads
        # nothing. Keying on the session reloaded ~14 GB at 1M for no reason.
        self.corpus_id:    str | None                = None
        self.generation:   str | None                = None
        self.chunks:       list[Chunk] | None        = None
        self.faiss_index:  faiss.Index | None        = None
        self.graph:        InMemoryGraphStore | None  = None
        self.pipeline:     QueryPipeline | None       = None
        # chunk_id -> position, built once per loaded index. Stage 2 used to
        # rebuild this for the whole corpus on every request.
        self.chunk_id_to_idx: dict[str, int] | None   = None
        # Inverted index published at build time, so no request rebuilds it.
        self.lexical = None
        self.questions = None
        self._lock = threading.Lock()
        self.loads = 0          # observability: how often we actually read from disk

    def _get_pipeline(self) -> QueryPipeline:
        if self.pipeline is None:
            self.pipeline = QueryPipeline(self._cfg)
        return self.pipeline

    def clear(self, corpus_id: str | None = None) -> None:
        """Drop the in-memory index. If a corpus is given, only clear when it matches."""
        with self._lock:
            if corpus_id is None or self.corpus_id == corpus_id:
                self.corpus_id = None
                self.generation = None
                self.chunks = None
                self.faiss_index = None
                self.graph = None
                self.chunk_id_to_idx = None
                self.lexical = None
                self.questions = None

    def ensure_loaded(self, corpus_id: str) -> bool:
        """Load a corpus's published generation, if not already resident.

        Returns True if a usable index is loaded, False if the corpus has none.
        A different session reading the SAME corpus generation is already served
        by what is resident, so it does no I/O at all.
        """
        with self._lock:
            gen = self._paths.generations(corpus_id).active()
            gen_id = gen[1].generation_id if gen else None
            if (self.corpus_id == corpus_id and self.generation == gen_id
                    and self.chunks is not None):
                return True
            if not self._paths.has_index(corpus_id):
                self.corpus_id = corpus_id
                self.generation = None
                self.chunks = None
                self.faiss_index = None
                self.graph = None
                self.chunk_id_to_idx = None
                self.lexical = None
                self.questions = None
                return False
            index_dir = self._paths.active_index_dir(corpus_id)
            if index_dir is None:
                return False
            store = IndexStore(index_dir, self._cfg.indexing)
            chunks, faiss_index = store.load()
            self.lexical = store.load_lexical()
            self.questions = store.load_questions(self._cfg.ollama.embedding_model)
            self.loads += 1
            graph = InMemoryGraphStore()
            graph.load(index_dir / self._paths.GRAPH_FILE)
            self.corpus_id = corpus_id
            self.generation = gen_id
            self.chunks = chunks
            self.faiss_index = faiss_index
            self.graph = graph
            self.chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(chunks)}
            self._get_pipeline()
            return True

    def query(self, question: str) -> QueryResult:
        """Run the full retrieval + generation pipeline against the active index."""
        return self._get_pipeline().query(
            question, self.chunks, self.faiss_index, self.graph,
            chunk_id_to_idx=self.chunk_id_to_idx,
            lexical=self.lexical,
            questions=self.questions,
        )

    def release(self, corpus_id: str) -> bool:
        """Drop this corpus if resident, so its files can be removed safely.

        Chunk embeddings are VIEWS into a memory-mapped vectors.npy. Unlinking
        that file while a reader holds the mapping leaves any page not already
        resident undefined — it happens to keep working on macOS, which is
        exactly how it would reach production unnoticed. Dropping the references
        closes the mapping first.
        """
        with self._lock:
            if self.corpus_id != corpus_id:
                return False
            self.corpus_id = self.generation = None
            self.chunks = self.faiss_index = self.graph = None
            self.chunk_id_to_idx = self.lexical = self.questions = None
            return True

    def close(self) -> None:
        if self.pipeline:
            self.pipeline.close()


class IndexingService:
    """Runs indexing for one session in a background thread (one job at a time)."""

    def __init__(
        self,
        cfg: Config,
        paths: ServerPaths,
        active: ActiveIndex,
        sessions: SessionStore,
    ) -> None:
        self._cfg = cfg
        self._paths = paths
        self._active = active
        self._sessions = sessions          # injected, not a module global
        self._lock = threading.Lock()
        self.running: bool       = False
        self.stage:   str        = ""
        self.error:   str | None = None
        self.session: str | None = None
        self.corpus:  str | None = None

    def is_building(self, corpus_id: str) -> bool:
        """True while a build is writing this corpus's directory."""
        with self._lock:
            return self.running and self.corpus == corpus_id

    @property
    def status(self) -> dict:
        return {
            "running": self.running,
            "stage":   self.stage,
            "error":   self.error,
            "session": self.session,
        }

    def start(self, sid: str) -> None:
        with self._lock:
            if self.running:
                raise RuntimeError("Indexing already in progress.")
            self.running = True
            self.error   = None
            self.stage   = "Starting…"
            self.session = sid
            self.corpus = self._sessions.corpus_of(sid) or sid
        threading.Thread(target=self._run, args=(sid,), daemon=True).start()

    def _set_stage(self, msg: str) -> None:
        with self._lock:
            self.stage = msg

    def _run(self, sid: str) -> None:
        # Indexing operates on the CORPUS the session reads, which may be shared
        # with other sessions. Publishing a generation makes it visible to all of
        # them at once.
        corpus_id = self._sessions.corpus_of(sid) or sid
        try:
            files = sorted(p for p in self._paths.docs_dir(corpus_id).glob("*")
                           if p.is_file())
            if not files:
                raise RuntimeError("No documents to index for this session.")

            # Build into a staging directory that no reader can see. The current
            # generation keeps serving throughout, and a build that dies part-way
            # leaves nothing published.
            gens = self._paths.generations(corpus_id)
            staged, gen_id = gens.stage()

            job_cfg = dataclasses.replace(self._cfg, store_path=staged)
            graph = InMemoryGraphStore()
            IndexingPipeline(job_cfg, graph).index(
                files, progress_cb=self._set_stage,
                doc_root=self._paths.docs_dir(corpus_id))
            graph.save(staged / self._paths.GRAPH_FILE)

            store = IndexStore(staged, self._cfg.indexing)
            count = store.chunk_count()
            if count is None:
                count = len(store.load()[0])

            self._set_stage("Publishing…")
            artifacts = store.artifacts()
            artifacts[self._paths.GRAPH_FILE] = -1
            gens.publish(staged, Manifest(
                generation_id=gen_id,
                created_at=time.time(),
                chunk_count=count,
                artifacts=artifacts,
                embedding_model=self._cfg.ollama.embedding_model,
                vector_index_type=self._cfg.indexing.vector_index_type,
                uq_prefix_role="query",
                # Frozen so a later incremental delta can be scored on the same
                # scale as this base; corpus-global extrema would otherwise
                # rescale every published edge when a new maximum arrives.
                score_transforms={
                    "edge_sparsify_threshold": self._cfg.indexing.edge_sparsify_threshold,
                    "shared_key_max_df_ratio": self._cfg.indexing.shared_key_max_df_ratio,
                    "shared_key_max_df_abs": self._cfg.indexing.shared_key_max_df_abs,
                    "shared_key_df_floor": self._cfg.indexing.shared_key_df_floor,
                },
            ))

            self._sessions.set_index_meta(
                sid, chunk_count=count, docs=[p.name for p in files])

            # Refresh the in-memory cache if this session is the active one.
            self._active.clear(corpus_id)
            self._active.ensure_loaded(corpus_id)
        except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
            logger.exception("Indexing failed for session %s", sid)
            self._paths.generations(corpus_id).discard_staging()
            with self._lock:
                self.error = str(exc)
        finally:
            with self._lock:
                self.running = False
                self.corpus  = None
                self.stage   = ""
                # Keep self.session pointing at this job so the UI can attribute
                # the result (or error) to the right session after it finishes.
