"""SpectRAG FastAPI server.

Class-based assembly: ``SpectRAGServer`` owns the shared config, the on-disk
paths, and the three stateful services (SessionStore, ActiveIndex,
IndexingService), and wires them into a FastAPI app whose routes are its own
methods. Import ``app`` (or run ``python server.py``) to serve it.
"""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from config import Config
from rag_server.graph_view import GraphView
from rag_server.runtime import ServerPaths
from rag_server.schemas import CreateSessionBody, QueryBody
from rag_server.services import ActiveIndex, IndexingService
from rag_server.session_store import SessionStore

logger = logging.getLogger(__name__)


class AccessLogFilter(logging.Filter):
    """Silences noisy polling requests from the access log."""

    def __init__(self, suppressed_paths: list[str]) -> None:
        super().__init__()
        self._suppressed = suppressed_paths

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(path in msg for path in self._suppressed)


class SpectRAGServer:
    """Owns the server's state and builds its FastAPI app."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self.paths = ServerPaths(self.cfg.store_path)
        self.sessions = SessionStore(self.paths, self.cfg.server.session_id_chars)
        self.active = ActiveIndex(self.cfg, self.paths)
        self.indexer = IndexingService(self.cfg, self.paths, self.active, self.sessions)
        self.graph_view = GraphView(self.cfg)

        logging.getLogger("uvicorn.access").addFilter(
            AccessLogFilter(self.cfg.server.log_suppress_paths))

        self.app = self._build_app()

    # App assembly

    def _build_app(self) -> FastAPI:
        app = FastAPI(
            title="SpectRAG — Spectral Graph-Augmented Retrieval",
            lifespan=self._lifespan,
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.cfg.server.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        add = app.add_api_route
        add("/api/status",                    self.get_status,      methods=["GET"])
        add("/api/sessions",                  self.list_sessions,   methods=["GET"])
        add("/api/sessions",                  self.create_session,  methods=["POST"])
        add("/api/sessions/{sid}",            self.get_session,     methods=["GET"])
        add("/api/sessions/{sid}",            self.rename_session,  methods=["PATCH"])
        add("/api/sessions/{sid}",            self.delete_session,  methods=["DELETE"])
        add("/api/sessions/{sid}/messages",   self.get_messages,    methods=["GET"])
        add("/api/sessions/{sid}/index",      self.index_documents, methods=["POST"])
        add("/api/sessions/{sid}/graph",      self.get_graph,       methods=["GET"])
        add("/api/sessions/{sid}/query",      self.query,           methods=["POST"])
        return app

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        self.paths.sessions_dir.mkdir(parents=True, exist_ok=True)
        logger.info("SpectRAG server ready — %d session(s) on disk.",
                    len(self.sessions.list_all()))
        yield
        self.active.close()

    def _require_session(self, sid: str) -> dict:
        """Resolve a session or raise 404 — shared by every /sessions/{sid}/… route."""
        session = self.sessions.get(sid)
        if session is None:
            raise HTTPException(404, "Session not found.")
        return session

    # Routes

    def get_status(self) -> dict:
        s = self.indexer.status
        return {
            "indexing":         s["running"],
            "indexing_stage":   s["stage"],
            "indexing_session": s["session"],
            "index_error":      s["error"],
        }

    def list_sessions(self) -> list[dict]:
        return self.sessions.list_all()

    def create_session(self, body: CreateSessionBody = None) -> dict:
        return self.sessions.create(body.name if body else None)

    def get_session(self, sid: str) -> dict:
        detail = self.sessions.detail(sid)
        if detail is None:
            raise HTTPException(404, "Session not found.")
        return detail

    def rename_session(self, sid: str, body: CreateSessionBody) -> dict:
        self._require_session(sid)
        name = (body.name or "").strip() if body else ""
        if not name:
            raise HTTPException(400, "A non-empty name is required.")
        self.sessions.rename(sid, name)
        return {"id": sid, "name": name}

    def delete_session(self, sid: str) -> dict:
        if not self.sessions.delete(sid):
            raise HTTPException(404, "Session not found.")
        self.active.clear(sid)
        return {"status": "deleted"}

    def get_messages(self, sid: str) -> list:
        return self._require_session(sid)["messages"]

    async def index_documents(self, sid: str, files: list[UploadFile] = File(...)) -> dict:
        self._require_session(sid)
        if self.indexer.running:
            raise HTTPException(409, "Indexing already in progress.")
        if not files:
            raise HTTPException(400, "No files provided.")

        # Save uploads into the session's permanent docs/ dir, then (re)index the
        # whole dir so new documents are ADDED to the session, not replacing it.
        docs = self.paths.docs_dir(sid)
        docs.mkdir(parents=True, exist_ok=True)
        for upload in files:
            name = Path(upload.filename or f"file_{uuid.uuid4().hex[:8]}").name
            (docs / name).write_bytes(await upload.read())

        self.indexer.start(sid)
        return {"status": "indexing_started", "file_count": len(files)}

    def get_graph(self, sid: str) -> dict:
        self._require_session(sid)
        if not self.active.ensure_loaded(sid) or self.active.graph is None:
            raise HTTPException(400, "This session has no index yet. Upload documents first.")
        return self.graph_view.build_payload(self.active)

    def query(self, sid: str, body: QueryBody) -> dict:
        self._require_session(sid)
        if self.indexer.running and self.indexer.session == sid:
            raise HTTPException(409, "Indexing in progress — please wait.")
        if not self.active.ensure_loaded(sid) or self.active.chunks is None:
            raise HTTPException(400, "This session has no index yet. Upload documents first.")

        result = self.active.query(body.question)

        sources = sorted(
            [
                {
                    "source":       c.metadata.get("source", c.doc_id),
                    "page":         c.metadata.get("page"),
                    "score":        round(result.scores.get(c.chunk_id, 0.0), 4),
                    "preview":      c.text[: self.cfg.server.source_preview_chars],
                    "full_text":    c.text,
                    "position":     c.position,
                    "section_path": c.section_path,
                }
                for c in result.chunks
            ],
            key=lambda s: s["score"],
            reverse=True,
        )

        self.sessions.append_exchange(sid, body.question, result.answer, sources)
        return {"answer": result.answer, "sources": sources}


# Module-level app so `uvicorn server:app` and `python server.py` both work.
server = SpectRAGServer()
app = server.app


if __name__ == "__main__":
    uvicorn.run("server:app", host=server.cfg.server.host,
                port=server.cfg.server.port, reload=False)
