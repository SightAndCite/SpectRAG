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
from config import OllamaConfig

logger = logging.getLogger(__name__)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class _EmbeddingCache:
    """Content-hash disk cache of chunk embeddings, one file per model.

    Keyed by the SHA-256 of the chunk text, so re-indexing a corpus after adding
    documents does not re-embed unchanged chunks. Deterministic: a hit returns the
    exact vector a fresh embed would produce, so results never change — only the
    recompute is skipped.
    """

    def __init__(self, cache_dir: str, model: str) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        # Model in the filename so switching embedders never serves stale vectors.
        self._file = self._dir / f"{model.replace('/', '_')}.pkl"
        self._cache: dict[str, np.ndarray] = {}
        if self._file.exists():
            try:
                with open(self._file, "rb") as fh:
                    self._cache = pickle.load(fh)
            except Exception:  # noqa: BLE001 — a corrupt cache must not abort indexing
                self._cache = {}
        self._dirty = False

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def get(self, key: str) -> np.ndarray | None:
        return self._cache.get(key)

    def put(self, key: str, vec: np.ndarray) -> None:
        self._cache[key] = np.asarray(vec, dtype=np.float32)
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        try:
            tmp = self._file.with_suffix(".pkl.tmp")
            with open(tmp, "wb") as fh:
                pickle.dump(self._cache, fh, protocol=5)
            tmp.replace(self._file)
            self._dirty = False
        except OSError:
            pass


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

    def embed(self, texts: list[str], kind: str = "document") -> np.ndarray:
        """Return (N, D) float32 array of L2-normalized embeddings.

        `kind` is "document" (default) or "query" — it selects the nomic task
        prefix. Use "query" for search queries; "document" for indexed content.
        """
        if not texts:
            return np.empty((0, self._dim or self.cfg.embedding_dim), dtype=np.float32)

        all_embeddings: list[list[float]] = []
        batch_size = self.cfg.embed_batch_size
        prefix = self._prefix(kind)

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            if prefix:
                batch = [prefix + t for t in batch]
            all_embeddings.extend(self._embed_batch(batch))
            logger.debug(
                "Embedded %d / %d texts",
                min(start + batch_size, len(texts)),
                len(texts),
            )

        arr = np.array(all_embeddings, dtype=np.float32)
        if arr.ndim != 2:
            raise RuntimeError(f"Malformed embeddings, expected 2-D got shape {arr.shape}")

        # Infer the true dimension from the model; warn once if it disagrees with
        # config, and fail fast on any later dimension drift (e.g. model swap).
        dim = arr.shape[1]
        if self._dim is None:
            self._dim = dim
            if dim != self.cfg.embedding_dim:
                logger.warning(
                    "Embedding dim %d differs from configured embedding_dim %d — "
                    "using the model's actual %d.", dim, self.cfg.embedding_dim, dim,
                )
        elif dim != self._dim:
            raise RuntimeError(
                f"Embedding dimension changed mid-run: {dim} != {self._dim}"
            )

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
            for pos, i in enumerate(miss_idx):
                self._emb_cache.put(keys[i], miss_embs[pos])
            self._emb_cache.flush()
            logger.info(
                "Embeddings: %d computed, %d reused from cache",
                len(miss_idx), len(chunks) - len(miss_idx),
            )
        else:
            logger.info("Embeddings: all %d reused from cache", len(chunks))

        for chunk, key in zip(chunks, keys):
            chunk.embedding = self._emb_cache.get(key)

    def build_faiss_index(self, chunks: list[Chunk]) -> faiss.IndexFlatIP:
        """Build an exact inner-product FAISS index from already-embedded chunks."""
        missing = [c.chunk_id for c in chunks if c.embedding is None]
        if missing:
            raise ValueError(
                f"{len(missing)} chunk(s) not embedded (e.g. {missing[:3]}) — "
                "call embed_chunks() before build_faiss_index()."
            )
        if not chunks:
            return faiss.IndexFlatIP(self._dim or self.cfg.embedding_dim)
        matrix = np.stack([c.embedding for c in chunks]).astype(np.float32)
        index = faiss.IndexFlatIP(matrix.shape[1])   # use the real embedding dim
        index.add(matrix)
        return index
