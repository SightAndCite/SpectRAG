from __future__ import annotations
import hashlib
import logging
import pickle
from pathlib import Path

import numpy as np
import faiss
import ollama
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
from rag_system.models import Chunk
from rag_system.indexing.concurrency import bounded_imap_batches
from rag_system.store.kv_cache import KeyValueCache, fingerprint
from rag_system.store.vector_index import build_vector_index
from config import OllamaConfig

logger = logging.getLogger(__name__)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class _EmbeddingCache:
    """Content-hash cache of chunk embeddings, backed by SQLite.

    Keyed by the SHA-256 of the chunk text, so re-indexing after adding documents
    does not re-embed unchanged chunks. Deterministic: a hit returns the exact
    vector a fresh embed would produce, so results never change — only the
    recompute is skipped.

    Previously one pickle dict, loaded whole and rewritten whole: 3.1 GB in memory
    at 1M chunks on every index run. Vectors are now stored as raw float32 bytes
    and read individually.
    """

    def __init__(self, cache_dir: str, model: str, dim: int | None = None) -> None:
        self._dir = Path(cache_dir)
        self._legacy = self._dir / f"{model.replace('/', '_')}.pkl"
        # Model in the namespace so switching embedders never serves stale vectors.
        self._cache = KeyValueCache(
            self._dir / "embeddings.sqlite3",
            fingerprint(kind="embedding", model=model),
        )
        self._cache.migrate_once(self._legacy, self._decode_legacy)

    @staticmethod
    def _decode_legacy(path: Path) -> dict[str, bytes]:
        with open(path, "rb") as fh:
            old = pickle.load(fh)
        return {k: np.asarray(v, dtype=np.float32).tobytes() for k, v in old.items()}

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def get(self, key: str) -> np.ndarray | None:
        raw = self._cache.get(key)
        return None if raw is None else np.frombuffer(raw, dtype=np.float32)

    def put(self, key: str, vec: np.ndarray) -> None:
        self._cache.put(key, np.asarray(vec, dtype=np.float32).tobytes())

    def put_many(self, items) -> None:
        self._cache.put_many(
            (k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in items)

    def flush(self) -> None:
        """No-op: every write is already committed."""

    def close(self) -> None:
        self._cache.close()


class OllamaEmbedder:
    """Batch embedder backed by Ollama's nomic-embed-text."""

    # nomic-embed-text is trained with task-instruction prefixes: documents must
    # be embedded as "search_document: ..." and queries as "search_query: ...".
    # Omitting them degrades retrieval and collapses the query/document
    # asymmetry. Only applied to nomic models; other embedders get no prefix.
    _QUERY_PREFIX = "search_query: "
    _DOCUMENT_PREFIX = "search_document: "

    def __init__(self, cfg: OllamaConfig) -> None:
        self.cfg = cfg
        # Apply the configured timeout so a stalled Ollama can't hang the run forever.
        self._client = ollama.Client(host=cfg.base_url, timeout=cfg.request_timeout)
        self._dim: int | None = None   # inferred from the first response
        self._emb_cache: _EmbeddingCache | None = None

    def _prefix(self, kind: str) -> str:
        """Task prefix for the current model ('' for non-nomic models)."""
        if "nomic" not in self.cfg.embedding_model.lower():
            return ""
        return self._QUERY_PREFIX if kind == "query" else self._DOCUMENT_PREFIX

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """One Ollama embed call, retried with backoff on transient failures."""
        resp = self._client.embed(model=self.cfg.embedding_model, input=batch)
        embs = resp.embeddings
        # Guard against silent misalignment: a short/partial response would map
        # the wrong vector onto a chunk. Treat it as retryable.
        if embs is None or len(embs) != len(batch):
            raise RuntimeError(
                f"Ollama returned {0 if embs is None else len(embs)} embeddings "
                f"for {len(batch)} inputs"
            )
        return embs

    def _check_dim(self, block: np.ndarray) -> None:
        """Validate a response block and pin the model's true dimension."""
        if block.ndim != 2:
            raise RuntimeError(f"Malformed embeddings, expected 2-D got shape {block.shape}")
        dim = block.shape[1]
        if self._dim is None:
            self._dim = dim
            if dim != self.cfg.embedding_dim:
                logger.warning(
                    "Embedding dim %d differs from configured embedding_dim %d — "
                    "using the model's actual %d.", dim, self.cfg.embedding_dim, dim,
                )
        elif dim != self._dim:
            raise RuntimeError(f"Embedding dimension changed mid-run: {dim} != {self._dim}")

    def embed(self, texts: list[str], kind: str = "document") -> np.ndarray:
        """Return (N, D) float32 array of L2-normalized embeddings.

        `kind` is "document" (default) or "query" — it selects the nomic task
        prefix. Use "query" for search queries; "document" for indexed content.
        """
        if not texts:
            return np.empty((0, self._dim or self.cfg.embedding_dim), dtype=np.float32)

        batch_size = self.cfg.embed_batch_size
        prefix = self._prefix(kind)

        # Fill a preallocated float32 array batch by batch. Accumulating the
        # response lists first kept every value alive as a Python float: a
        # 1M-chunk pass is 768M of them, tens of GB, before np.array() ever ran.
        # Only one batch of Python floats is live at a time now.
        def _run(start: int, batch: list[str]) -> np.ndarray:
            if prefix:
                batch = [prefix + x for x in batch]
            return np.asarray(self._embed_batch(batch), dtype=np.float32)

        # The first batch runs alone to establish the true dimension and size the
        # output array; the rest overlap. Each result carries its own offset, so
        # ordering is structural rather than something to reassemble.
        head = _run(0, texts[:batch_size])
        self._check_dim(head)
        arr = np.empty((len(texts), head.shape[1]), dtype=np.float32)
        arr[: head.shape[0]] = head

        rest = texts[batch_size:]
        if rest:
            done = head.shape[0]
            for offset, block in bounded_imap_batches(
                _run, rest, batch_size=batch_size,
                workers=max(1, self.cfg.embed_max_concurrency),
            ):
                self._check_dim(block)
                start = batch_size + offset
                arr[start : start + block.shape[0]] = block
                done += block.shape[0]
                logger.debug("Embedded %d / %d texts", done, len(texts))

        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return arr / norms

    def embed_chunks(self, chunks: list[Chunk]) -> None:
        """Embed all chunks in-place, reusing cached vectors for unchanged text.

        Only chunks whose text is not already in the content-hash cache are sent to
        Ollama; the rest are served from disk. This makes re-indexing after adding
        documents pay embedding cost only for genuinely new chunks.
        """
        if not chunks:
            return
        if self._emb_cache is None:
            self._emb_cache = _EmbeddingCache(
                self.cfg.embedding_cache_dir, self.cfg.embedding_model
            )

        keys = [_text_hash(c.text) for c in chunks]
        miss_idx = [i for i, k in enumerate(keys) if k not in self._emb_cache]

        if miss_idx:
            miss_embs = self.embed([chunks[i].text for i in miss_idx], kind="document")
            self._emb_cache.put_many(
                (keys[i], miss_embs[pos]) for pos, i in enumerate(miss_idx))
            logger.info(
                "Embeddings: %d computed, %d reused from cache",
                len(miss_idx), len(chunks) - len(miss_idx),
            )
        else:
            logger.info("Embeddings: all %d reused from cache", len(chunks))

        for chunk, key in zip(chunks, keys):
            chunk.embedding = self._emb_cache.get(key)

    def build_faiss_index(self, chunks: list[Chunk], *, index_type: str = "flat",
                          m: int = 32, ef_construction: int = 200,
                          ef_search: int = 128) -> faiss.Index:
        """Build the chunk search index from already-embedded chunks."""
        missing = [c.chunk_id for c in chunks if c.embedding is None]
        if missing:
            raise ValueError(
                f"{len(missing)} chunk(s) not embedded (e.g. {missing[:3]}) — "
                "call embed_chunks() before build_faiss_index()."
            )
        if not chunks:
            return faiss.IndexFlatIP(self._dim or self.cfg.embedding_dim)
        matrix = np.stack([c.embedding for c in chunks]).astype(np.float32)
        return build_vector_index(matrix, index_type=index_type, m=m,
                                  ef_construction=ef_construction, ef_search=ef_search)
