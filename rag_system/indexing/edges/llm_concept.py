"""LLM concept extraction for the entity/concept edge signal.

Instead of spaCy/GLiNER named entities (generic persons/orgs, weakly extracted),
this asks an LLM (gpt-4o-mini) for the domain CONCEPTS a passage is about, as
canonical short phrases. Two chunks that discuss the same concept in different
words — which cosine similarity misses — then share a concept key and get linked.
This is the cross-document signal genuinely ORTHOGONAL to the embedding space,
which is the graph's whole reason to exist.

Results are cached on disk by content hash, so re-indexing the same corpus (the
per-domain eval loop does this a lot) is free after the first pass. Failed calls
retry with exponential backoff and are NEVER cached — a transient 429/5xx must
not permanently strip a chunk of its concepts. extract_batch() runs the calls in
a thread pool (llm_parallel_workers) to cut indexing wall-clock.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from config import IndexingConfig, OpenAIConfig
from prompts import CONCEPT_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class LLMConceptExtractor:
    def __init__(self, openai_cfg: OpenAIConfig, cfg: IndexingConfig) -> None:
        if not openai_cfg.api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for entity_extraction_mode='llm_concepts'."
            )
        self._client = OpenAI(
            api_key=openai_cfg.api_key,
            base_url=openai_cfg.base_url,
            timeout=openai_cfg.request_timeout,
        )
        self._model = openai_cfg.model
        self._max_chars = cfg.concept_max_text_chars
        self._max_tokens = cfg.concept_llm_max_tokens
        self._workers = max(1, cfg.llm_parallel_workers)
        self._max_retries = max(1, cfg.llm_max_retries)

        self._cache_dir = Path(cfg.concept_cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self._cache_dir / f"{self._model}.json"
        self._cache: dict[str, list[str]] = {}
        if self._cache_file.exists():
            try:
                self._cache = json.loads(self._cache_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._cache = {}
        self._dirty = 0
        self._lock = threading.Lock()  # guards _cache/_dirty across pool threads

    def _call_llm(self, snippet: str) -> list[str]:
        """One extraction call with retry/backoff. Raises after final attempt."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user",
                               "content": CONCEPT_EXTRACTION_PROMPT.format(text=snippet)}],
                    temperature=0.0,
                    max_tokens=self._max_tokens,
                    response_format={"type": "json_object"},
                )
                data = json.loads(resp.choices[0].message.content or "{}")
                return sorted({
                    str(c).strip().lower() for c in data.get("concepts", []) if str(c).strip()
                })
            except Exception as exc:  # noqa: BLE001 — retry transient API/parse errors
                last_exc = exc
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)  # 1s, 2s, …
        raise last_exc  # type: ignore[misc]

    def extract(self, text: str) -> list[str]:
        """Concept keys for one chunk (cached by content hash; failures NOT cached)."""
        snippet = text[: self._max_chars]
        key = hashlib.sha256(snippet.encode("utf-8")).hexdigest()[:16]
        with self._lock:
            if key in self._cache:
                return self._cache[key]

        try:
            concepts = self._call_llm(snippet)
        except Exception as exc:  # noqa: BLE001 — one bad chunk must not abort indexing
            logger.warning("Concept extraction failed after %d attempts (%s); "
                           "no concepts for this chunk (not cached — retried next run)",
                           self._max_retries, exc)
            return []

        with self._lock:
            self._cache[key] = concepts
            self._dirty += 1
            if self._dirty >= 25:
                self._flush_locked()
        return concepts

    def extract_batch(self, texts: list[str],
                      progress_cb=None, progress_every: int = 50) -> list[list[str]]:
        """Concepts for many chunks, LLM calls in parallel. Order preserved."""
        results: list[list[str]] = [[] for _ in texts]
        done = 0
        done_lock = threading.Lock()

        def _one(i: int) -> None:
            nonlocal done
            results[i] = self.extract(texts[i])
            if progress_cb:
                with done_lock:
                    done += 1
                    if done % progress_every == 0:
                        progress_cb(done, len(texts))

        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            list(pool.map(_one, range(len(texts))))
        return results

    def _flush_locked(self) -> None:
        """Write the cache to disk. Caller must hold self._lock."""
        try:
            self._cache_file.write_text(json.dumps(self._cache), encoding="utf-8")
            self._dirty = 0
        except OSError:
            pass

    def close(self) -> None:
        with self._lock:
            self._flush_locked()
