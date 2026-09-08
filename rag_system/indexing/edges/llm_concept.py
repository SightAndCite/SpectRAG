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
from pathlib import Path

from openai import OpenAI

from rag_system.indexing.concurrency import bounded_map
from rag_system.store.kv_cache import KeyValueCache, fingerprint
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
        self._in_flight = max(self._workers, cfg.llm_max_in_flight)
        self._max_retries = max(1, cfg.llm_max_retries)

        # Transactional keyed store. The previous JSON dict was rewritten whole
        # every 25 additions while holding the lock every worker needed, so
        # cumulative I/O grew quadratically (~4 TB to fill 1M entries) and a
        # crash discarded everything since the last flush.
        self._cache_dir = Path(cfg.concept_cache_dir)
        self._legacy = self._cache_dir / f"{self._model}.json"
        self._cache = KeyValueCache(
            self._cache_dir / "concepts.sqlite3",
            # Prompt and generation settings belong in the key: changing either
            # previously kept serving results produced under the old ones.
            fingerprint(kind="concept", model=self._model,
                        prompt=CONCEPT_EXTRACTION_PROMPT,
                        max_chars=self._max_chars, max_tokens=self._max_tokens,
                        temperature=0.0),
        )
        self._cache.migrate_once(
            self._legacy,
            lambda pth: {k: json.dumps(v).encode("utf-8")
                         for k, v in json.loads(pth.read_text(encoding="utf-8")).items()},
        )

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
        hit = self._cache.get(key)
        if hit is not None:
            return json.loads(hit)

        try:
            concepts = self._call_llm(snippet)
        except Exception as exc:  # noqa: BLE001 — one bad chunk must not abort indexing
            logger.warning("Concept extraction failed after %d attempts (%s); "
                           "no concepts for this chunk (not cached — retried next run)",
                           self._max_retries, exc)
            return []

        self._cache.put(key, json.dumps(concepts).encode("utf-8"))
        return concepts

    def extract_batch(self, texts: list[str],
                      progress_cb=None, progress_every: int = 50) -> list[list[str]]:
        """Concepts for many chunks, LLM calls bounded and concurrent.

        Order preserved. ThreadPoolExecutor.map submitted the whole corpus, which
        is one Future per chunk before any work begins; this keeps a fixed number
        alive instead.
        """
        results, self.last_outcome = bounded_map(
            self.extract, texts,
            workers=self._workers, in_flight=self._in_flight,
            stage="concept extraction",
            on_progress=progress_cb, progress_every=progress_every,
        )
        return [r if r is not None else [] for r in results]

    def close(self) -> None:
        self._cache.close()
