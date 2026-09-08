"""Transactional keyed cache for index-time inference results.

The concept, utility-question and embedding caches all persisted by rewriting
their entire contents — the JSON ones every 25 additions, while holding the lock
that every extraction worker needs. Cumulative I/O therefore grows as
O(n²/flush): measured by extrapolation from 4,000 entries, filling 1M writes
about 4 TB, which on its own can exceed a 24-hour build budget. The embedding
cache additionally loaded a 3.1 GB pickle into memory on every index run, and
the concept cache's write was not atomic, so a crash mid-write corrupted it.

SQLite gives O(1) transactional writes per entry, so cumulative I/O is linear
and nothing committed is lost on a crash. It is in the standard library, and WAL
mode allows concurrent readers, which the previous instance-local locks did not.

Entries are namespaced by a dependency fingerprint — model revision, prompt
version, and the generation settings that shape the output. Keying on input text
alone meant that changing a prompt or model silently served results produced
under the old configuration. A changed dependency now lands in a new namespace,
so old entries are neither served nor destroyed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Callable, Iterable

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    ns    TEXT NOT NULL,
    key   TEXT NOT NULL,
    value BLOB NOT NULL,
    PRIMARY KEY (ns, key)
) WITHOUT ROWID;
"""


def fingerprint(**parts) -> str:
    """Short stable hash of the settings an entry's value depends on."""
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class KeyValueCache:
    """SQLite-backed cache of bytes, namespaced by dependency fingerprint."""

    def __init__(self, path: Path | str, namespace: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ns = namespace
        self._lock = threading.Lock()
        # Shared across the extractor thread pool; every access holds the lock,
        # and each one is a single indexed statement rather than a full rewrite.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM entries WHERE ns=? AND key=?", (self.ns, key)
            ).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: bytes) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO entries (ns, key, value) VALUES (?,?,?)",
                (self.ns, key, value),
            )
            self._conn.commit()

    def put_many(self, items: Iterable[tuple[str, bytes]]) -> int:
        rows = [(self.ns, k, v) for k, v in items]
        if not rows:
            return 0
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO entries (ns, key, value) VALUES (?,?,?)", rows
            )
            self._conn.commit()
        return len(rows)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT count(*) FROM entries WHERE ns=?", (self.ns,)
            ).fetchone()[0]

    def migrate_once(self, legacy: Path, decode: Callable[[Path], dict[str, bytes]]) -> int:
        """Import a legacy whole-file cache, once, leaving the original in place.

        Only runs when this namespace is empty, so it never re-imports and never
        overwrites newer entries. The legacy file is not deleted — it stays as a
        fallback until you remove it deliberately.

        Caveat worth knowing: legacy entries carry no record of the settings that
        produced them, so importing them asserts they came from the current
        configuration. That is the price of not re-spending on 2M LLM calls.
        """
        if not legacy.exists() or len(self) > 0:
            return 0
        try:
            items = decode(legacy)
        except Exception as exc:  # noqa: BLE001 — a corrupt legacy cache must not block a build
            logger.warning("Could not read legacy cache %s (%s) — starting empty", legacy, exc)
            return 0
        n = self.put_many(items.items())
        logger.info("Migrated %d entries from %s into %s (original kept)",
                    n, legacy.name, self.path.name)
        return n

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
                self._conn.close()
            except sqlite3.Error:
                pass
