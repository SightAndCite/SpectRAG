"""Server services: the active-session index cache and the indexing worker."""
from __future__ import annotations

import dataclasses
import logging
import threading
from typing import TYPE_CHECKING

from rag_system.indexing.pipeline import IndexingPipeline
from rag_system.query.pipeline import QueryPipeline
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
        self.sid:          str | None                = None
        self.chunks:       list[Chunk] | None        = None
        self.faiss_index:  faiss.Index | None        = None
        self.graph:        InMemoryGraphStore | None  = None
        self.pipeline:     QueryPipeline | None       = None
        self._lock = threading.Lock()

    def _get_pipeline(self) -> QueryPipeline:
        if self.pipeline is None:
            self.pipeline = QueryPipeline(self._cfg)
        return self.pipeline

    def clear(self, sid: str | None = None) -> None:
        """Drop the in-memory index. If sid is given, only clear when it matches."""
        with self._lock:
            if sid is None or self.sid == sid:
                self.sid = None
                self.chunks = None
                self.faiss_index = None
                self.graph = None

    def ensure_loaded(self, sid: str) -> bool:
        """Load ``sid``'s index into memory (from disk) if not already active.
        Returns True if a usable index is loaded, False if the session has none."""
        with self._lock:
            if self.sid == sid and self.chunks is not None:
                return True
            if not self._paths.has_index(sid):
                self.sid = sid
                self.chunks = None
                self.faiss_index = None
                self.graph = None
                return False
            chunks, faiss_index = IndexStore(self._paths.session_dir(sid)).load()
            graph = InMemoryGraphStore()
            graph.load(self._paths.graph_file(sid))
            self.sid = sid
            self.chunks = chunks
            self.faiss_index = faiss_index
            self.graph = graph
            self._get_pipeline()
            return True

    def query(self, question: str) -> QueryResult:
        """Run the full retrieval + generation pipeline against the active index."""
        return self._get_pipeline().query(
            question, self.chunks, self.faiss_index, self.graph
        )

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
        threading.Thread(target=self._run, args=(sid,), daemon=True).start()

    def _set_stage(self, msg: str) -> None:
        with self._lock:
            self.stage = msg

    def _run(self, sid: str) -> None:
        try:
            files = sorted(p for p in self._paths.docs_dir(sid).glob("*") if p.is_file())
            if not files:
                raise RuntimeError("No documents to index for this session.")

            # Per-job config that writes the index into this session's dir — a
            # copy (nested configs shared), so the shared cfg is never mutated.
            job_cfg = dataclasses.replace(self._cfg, store_path=self._paths.session_dir(sid))
            graph = InMemoryGraphStore()
            IndexingPipeline(job_cfg, graph).index(files, progress_cb=self._set_stage)
            graph.save(self._paths.graph_file(sid))

            chunks, _ = IndexStore(self._paths.session_dir(sid)).load()
            self._sessions.set_index_meta(
                sid, chunk_count=len(chunks), docs=[p.name for p in files])

            # Refresh the in-memory cache if this session is the active one.
            self._active.clear(sid)
            self._active.ensure_loaded(sid)
        except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
            logger.exception("Indexing failed for session %s", sid)
            with self._lock:
                self.error = str(exc)
        finally:
            with self._lock:
                self.running = False
                self.stage   = ""
                # Keep self.session pointing at this job so the UI can attribute
                # the result (or error) to the right session after it finishes.
