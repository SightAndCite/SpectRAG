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

    def has_index(self, sid: str) -> bool:
        """True once a session has been indexed (its chunk store exists on disk)."""
        return (self.session_dir(sid) / self.CHUNKS_FILE).exists()
