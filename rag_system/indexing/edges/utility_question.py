from __future__ import annotations
import hashlib
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np
import faiss
import ollama
from rag_system.models import Chunk
from rag_system.indexing.edges.base import EdgeExtractor, RawEdge
from rag_system.indexing.embedder import OllamaEmbedder
from rag_system.language.detector import language_name
from config import IndexingConfig, OllamaConfig, OpenAIConfig
from rag_system.indexing.concurrency import bounded_map
from rag_system.store.kv_cache import KeyValueCache, fingerprint
from prompts import UTILITY_QUESTION_PROMPT

logger = logging.getLogger(__name__)


class UtilityQuestionEdgeExtractor(EdgeExtractor):
    """
    Functional-similarity edges: the LLM generates questions per chunk; chunks
    that answer similar questions receive an edge.
    """

    def __init__(
        self,
        cfg: IndexingConfig,
        ollama_cfg: OllamaConfig,
        embedder: OllamaEmbedder,
        openai_cfg: OpenAIConfig | None = None,
    ) -> None:
        self._n_questions    = cfg.utility_questions_per_chunk
        self._threshold      = cfg.utility_question_threshold
        self._search_k       = cfg.utility_question_search_k
        self._max_text_chars = cfg.utility_question_max_text_chars
        self._temperature    = cfg.utility_question_llm_temperature
        self._max_tokens     = cfg.utility_question_llm_max_tokens
        self._log_interval   = cfg.utility_question_log_interval
        self._ollama_cfg     = ollama_cfg
        self._embedder       = embedder
        self._backend        = cfg.utility_question_llm_backend
        # OpenAI calls parallelize well; the local Ollama server is kept
        # sequential (it serializes generations anyway).
        self._workers        = max(1, cfg.llm_parallel_workers) if self._backend == "openai" else 1
        self._max_retries    = max(1, cfg.llm_max_retries)
        self._in_flight      = max(self._workers, cfg.llm_max_in_flight)
        if self._backend == "openai":
            if openai_cfg is None or not openai_cfg.api_key:
                raise ValueError("OPENAI_API_KEY required for utility_question_llm_backend='openai'.")
            from openai import OpenAI
            self._openai = OpenAI(
                api_key=openai_cfg.api_key, base_url=openai_cfg.base_url,
                timeout=openai_cfg.request_timeout,
            )
            self._openai_model = openai_cfg.model
            self._client = None
            model_tag = openai_cfg.model
        else:
            self._openai = None
            self._client = ollama.Client(host=ollama_cfg.base_url)
            model_tag = ollama_cfg.llm_model

        # Content-hash disk cache of generated questions, keyed by chunk text (+
        # language). The filename encodes backend/model/count so a config change
        # never serves mismatched questions. Re-indexing after adding docs only
        # pays the LLM for genuinely new chunks; unchanged chunks are free.
        # Transactional keyed store; see llm_concept.py for why the whole-file
        # JSON cache had to go.
        self._cache_dir = Path(cfg.utility_question_cache_dir)
        self._legacy = self._cache_dir / (
            f"{self._backend}_{model_tag.replace('/', '_')}_{self._n_questions}q.json"
        )
        self._cache = KeyValueCache(
            self._cache_dir / "questions.sqlite3",
            fingerprint(kind="utility_question", backend=self._backend, model=model_tag,
                        prompt=UTILITY_QUESTION_PROMPT, n=self._n_questions,
                        max_chars=self._max_text_chars, max_tokens=self._max_tokens,
                        temperature=self._temperature),
        )
        self._cache.migrate_once(
            self._legacy,
            lambda pth: {k: json.dumps(v).encode("utf-8")
                         for k, v in json.loads(pth.read_text(encoding="utf-8")).items()},
        )

    def extract(self, chunks: list[Chunk]) -> list[RawEdge]:
        # Generate questions for each chunk — LLM calls in parallel (OpenAI
        # backend), each with retry/backoff so transient API errors don't
        # silently strip a chunk of its questions.
        def _one(chunk) -> list[str]:
            return self._generate_questions(chunk.text, language_name(chunk.language))

        results, self.last_outcome = bounded_map(
            _one, chunks,
            workers=self._workers, in_flight=self._in_flight,
            stage="utility questions",
            on_progress=lambda d, n: logger.info("Utility questions: %d / %d chunks", d, n),
            progress_every=self._log_interval,
        )
        all_questions: list[list[str]] = [r if r is not None else [] for r in results]

        self._flush_cache()

        # Persist the questions on their chunk — reused at query time as
        # query-shaped surrogates for retrieval (see SeedRetriever).
        for ci, qs in enumerate(all_questions):
            chunks[ci].utility_questions = qs

        # Flatten, tracking which chunk each question belongs to
        flat_questions: list[str] = []
        q_to_chunk: list[int] = []
        for ci, qs in enumerate(all_questions):
            for q in qs:
                flat_questions.append(q)
                q_to_chunk.append(ci)

        if not flat_questions:
            return []

        # Embed all questions in one batched pass
        q_embeddings = self._embedder.embed(flat_questions)

        # ANN search over question embeddings
        k = min(self._search_k, len(flat_questions))
        index = faiss.IndexFlatIP(q_embeddings.shape[1])
        index.add(q_embeddings)
        scores_mat, idx_mat = index.search(q_embeddings, k)

        edges: dict[tuple[int, int], float] = {}
        for qi, (nbrs, sims) in enumerate(zip(idx_mat, scores_mat)):
            ci = q_to_chunk[qi]
            for qj, sim in zip(nbrs, sims):
                cj = q_to_chunk[qj]
                if ci == cj or sim < self._threshold:
                    continue
                key = (min(ci, cj), max(ci, cj))
                if sim > edges.get(key, -1.0):
                    edges[key] = float(sim)

        return [RawEdge(i, j, s) for (i, j), s in edges.items()]

    @staticmethod
    def _parse_questions(raw: str, n: int) -> list[str]:
        """Split model output into questions, stripping list numbering/bullets."""
        import re
        out: list[str] = []
        for ln in raw.strip().splitlines():
            ln = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", ln.strip())
            if ln:
                out.append(ln)
        return out[:n]

    def _generate_questions(self, text: str, language: str = "English") -> list[str]:
        snippet = text[: self._max_text_chars]
        # Cache key = language + text snippet (questions depend on both). A hit
        # returns without any LLM call.
        key = hashlib.sha256(f"{language}|{snippet}".encode("utf-8")).hexdigest()[:16]
        hit = self._cache.get(key)
        if hit is not None:
            return list(json.loads(hit))

        prompt = UTILITY_QUESTION_PROMPT.format(
            n=self._n_questions,
            language=language,
            text=snippet,
        )
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                if self._openai is not None:
                    resp = self._openai.chat.completions.create(
                        model=self._openai_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                    )
                    raw = resp.choices[0].message.content or ""
                else:
                    resp = self._client.generate(
                        model=self._ollama_cfg.llm_model,
                        prompt=prompt,
                        options={"temperature": self._temperature, "num_predict": self._max_tokens},
                    )
                    raw = resp.response
                questions = self._parse_questions(raw, self._n_questions)
                self._cache.put(key, json.dumps(questions).encode("utf-8"))
                return questions
            except Exception as exc:  # noqa: BLE001 — retry transient API errors
                last_exc = exc
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)  # 1s, 2s, …
        # Failures are NOT cached — a transient error must not permanently strip a
        # chunk of its questions (it retries on the next index run).
        logger.warning("Question generation failed after %d attempts: %s",
                       self._max_retries, last_exc)
        return []

    def _flush_cache(self) -> None:
        """No-op: every write is already committed."""
