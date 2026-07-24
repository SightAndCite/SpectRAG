"""Persisted chat sessions.

Sessions are stored in ``sessions.json`` so they survive restarts, and each
session owns its documents + index under ``sessions/<id>/`` — so reopening a
session (or restarting the server) restores everything it had.
"""
from __future__ import annotations

import json
import shutil
import threading
import uuid
from datetime import datetime, timezone

from rag_server.runtime import ServerPaths


class SessionStore:
    """Thread-safe, disk-persisted store of chat sessions."""

    def __init__(self, paths: ServerPaths, id_chars: int) -> None:
        self._paths = paths
        self._id_chars = id_chars
        self._lock = threading.Lock()
        self._sessions: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self._paths.sessions_file.exists():
            try:
                data = json.loads(self._paths.sessions_file.read_text(encoding="utf-8"))
                self._sessions = {s["id"]: s for s in data}
            except (json.JSONDecodeError, OSError, KeyError):
                self._sessions = {}

    def _save(self) -> None:
        path = self._paths.sessions_file
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(list(self._sessions.values()), indent=2), encoding="utf-8")
        tmp.replace(path)  # atomic — no torn writes

    def create(self, name: str | None = None) -> dict:
        with self._lock:
            sid = str(uuid.uuid4())[: self._id_chars]
            name = (name or "").strip() or f"Session {len(self._sessions) + 1}"
            self._sessions[sid] = {
                "id":          sid,
                "name":        name,
                "messages":    [],
                "created_at":  datetime.now(timezone.utc).isoformat(),
                "chunk_count": 0,
                "docs":        [],
            }
            self._save()
        self._paths.docs_dir(sid).mkdir(parents=True, exist_ok=True)
        return {"id": sid, "name": name}

    def _summary(self, d: dict) -> dict:
        return {
            "id":            d["id"],
            "name":          d["name"],
            "message_count": len(d["messages"]),
            "created_at":    d["created_at"],
            "chunk_count":   d.get("chunk_count", 0),
            "doc_count":     len(d.get("docs", [])),
            "docs":          d.get("docs", []),
            "has_index":     self._paths.has_index(d["id"]),
        }

    def list_all(self) -> list[dict]:
        return [self._summary(d) for d in self._sessions.values()]

    def get(self, sid: str) -> dict | None:
        return self._sessions.get(sid)

    def detail(self, sid: str) -> dict | None:
        d = self._sessions.get(sid)
        return self._summary(d) if d else None

    def rename(self, sid: str, name: str) -> bool:
        name = (name or "").strip()
        if not name:
            return False
        with self._lock:
            if sid not in self._sessions:
                return False
            self._sessions[sid]["name"] = name
            self._save()
        return True

    def delete(self, sid: str) -> bool:
        with self._lock:
            if sid not in self._sessions:
                return False
            del self._sessions[sid]
            self._save()
        shutil.rmtree(self._paths.session_dir(sid), ignore_errors=True)
        return True

    def set_index_meta(self, sid: str, chunk_count: int, docs: list[str]) -> None:
        with self._lock:
            if sid in self._sessions:
                self._sessions[sid]["chunk_count"] = chunk_count
                self._sessions[sid]["docs"] = docs
                self._save()

    def append_exchange(self, sid: str, question: str, answer: str, sources: list) -> None:
        with self._lock:
            session = self._sessions[sid]
            session["messages"].append({"role": "user", "content": question})
            session["messages"].append(
                {"role": "assistant", "content": answer, "sources": sources})
            self._save()
