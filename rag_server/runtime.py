"""On-disk layout for the server, as a class.

Every session owns its documents and index under ``<store_dir>/sessions/<id>/``,
so reopening a session (or restarting the server) restores everything it had:

    sessions.json                     session metadata (name, messages, …)
    sessions/<id>/docs/               the session's uploaded source files
    sessions/<id>/chunks.pkl
    sessions/<id>/faiss.index
    sessions/<id>/graph.pkl           that session's chunk graph
"""
from __future__ import annotations

from pathlib import Path

from rag_system.store.generation import GenerationStore


class ServerPaths:
    """Resolves every on-disk path the server uses, from a single store dir."""

    GRAPH_FILE = "graph.pkl"
    CHUNKS_FILE = "chunks.pkl"

    def __init__(self, store_dir: Path | str) -> None:
        self.store_dir = Path(store_dir)
        self.sessions_dir = self.store_dir / "sessions"
        self.sessions_file = self.store_dir / "sessions.json"

    def session_dir(self, sid: str) -> Path:
        return self.sessions_dir / sid

    def docs_dir(self, sid: str) -> Path:
        return self.session_dir(sid) / "docs"

    def graph_file(self, sid: str) -> Path:
        return self.session_dir(sid) / self.GRAPH_FILE

    def generations(self, sid: str) -> GenerationStore:
        return GenerationStore(self.session_dir(sid))

    def active_index_dir(self, sid: str) -> Path | None:
        """Directory holding this session's published index, or None.

        Resolves the generation manifest; callers should resolve once and reuse
        the path, so a publication mid-request cannot move files underneath them.
        Falls back to the flat legacy layout for indexes built before generations.
        """
        act = self.generations(sid).active()
        if act is not None:
            return act[0]
        legacy = self.session_dir(sid)
        return legacy if (legacy / self.CHUNKS_FILE).exists() else None

    def has_index(self, sid: str) -> bool:
        """True once a session has a COMPLETE published index.

        Previously this tested only for chunks.pkl, so a build interrupted
        between artifact writes read as ready.
        """
        return self.active_index_dir(sid) is not None
