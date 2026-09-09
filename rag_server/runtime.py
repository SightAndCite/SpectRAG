"""On-disk layout for the server, as a class.

Documents and index generations belong to a CORPUS, not to a chat session:

    sessions.json                            session metadata, incl. corpus_id
    corpora/<corpus_id>/docs/                the uploaded source files
    corpora/<corpus_id>/manifest.json        pointer to the active generation
    corpora/<corpus_id>/generations/<id>/    one complete, immutable index

Everything used to live under ``sessions/<sid>/``, which meant two chats over the
same documents each held their own copy and each paid for their own build. At 1M
chunks a corpus is roughly 14 GB of artifacts, so five such chats is ~71 GB and
five separate embedding and extraction runs. Worse, the in-memory cache was keyed
by session, so switching between two chats over identical documents evicted and
reloaded the whole index.

A session now records which corpus it reads. Creating a chat still creates a
private corpus, so nothing about the existing flow changes; the difference is
that a second session *can* reference the same one, and the cache key is the
corpus generation rather than the chat.

Sessions indexed under the old flat layout keep working: `corpus_dir` falls back
to the legacy session directory when no corpus directory exists.
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
        self.corpora_dir = self.store_dir / "corpora"
        self.sessions_file = self.store_dir / "sessions.json"

    # Sessions

    def session_dir(self, sid: str) -> Path:
        return self.sessions_dir / sid

    # Corpora

    def corpus_dir(self, corpus_id: str) -> Path:
        """Where a corpus keeps its documents and generations.

        Falls back to the legacy per-session directory when a corpus of that id
        has none of its own, so indexes built before corpora existed still
        resolve without a rebuild.
        """
        new = self.corpora_dir / corpus_id
        if new.exists():
            return new
        legacy = self.sessions_dir / corpus_id
        if legacy.exists():
            return legacy
        return new

    def docs_dir(self, corpus_id: str) -> Path:
        return self.corpus_dir(corpus_id) / "docs"

    def generations(self, corpus_id: str) -> GenerationStore:
        return GenerationStore(self.corpus_dir(corpus_id))

    def active_index_dir(self, corpus_id: str) -> Path | None:
        """Directory holding this corpus's published index, or None.

        Resolves the generation manifest; callers should resolve once and reuse
        the path, so a publication mid-request cannot move files underneath them.
        Falls back to the flat legacy layout for indexes built before generations.
        """
        act = self.generations(corpus_id).active()
        if act is not None:
            return act[0]
        legacy = self.corpus_dir(corpus_id)
        return legacy if (legacy / self.CHUNKS_FILE).exists() else None

    def has_index(self, corpus_id: str) -> bool:
        """True once a corpus has a COMPLETE published index.

        Previously this tested only for chunks.pkl, so a build interrupted
        between artifact writes read as ready.
        """
        return self.active_index_dir(corpus_id) is not None
