from __future__ import annotations
import hashlib
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import faiss
import ollama
from rag_system.models import Chunk
from rag_system.indexing.edges.base import EdgeExtractor, RawEdge
from rag_system.indexing.embedder import OllamaEmbedder
from rag_system.language.detector import language_name
from config import IndexingConfig, OllamaConfig, OpenAIConfig
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
        self._cache_dir = Path(cfg.utility_question_cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self._cache_dir / (
            f"{self._backend}_{model_tag.replace('/', '_')}_{self._n_questions}q.json"
        )
        self._cache: dict[str, list[str]] = {}
        if self._cache_file.exists():
            try:
                self._cache = json.loads(self._cache_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._cache = {}
        self._cache_lock = threading.Lock()
        self._cache_dirty = 0

    def extract(self, chunks: list[Chunk]) -> list[RawEdge]:
        # Generate questions for each chunk — LLM calls in parallel (OpenAI
        # backend), each with retry/backoff so transient API errors don't
        # silently strip a chunk of its questions.
        all_questions: list[list[str]] = [[] for _ in chunks]
        done = 0
        done_lock = threading.Lock()

        def _one(i: int) -> None:
            nonlocal done
            all_questions[i] = self._generate_questions(
                chunks[i].text, language_name(chunks[i].language))
            with done_lock:
                done += 1
                if done % self._log_interval == 0:
                    logger.info("Utility questions: %d / %d chunks", done, len(chunks))

        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            list(pool.map(_one, range(len(chunks))))

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
        with self._cache_lock:
            if key in self._cache:
                return list(self._cache[key])

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
                with self._cache_lock:
                    self._cache[key] = questions
                    self._cache_dirty += 1
                    if self._cache_dirty >= 25:
                        self._flush_cache_locked()
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

    def _flush_cache_locked(self) -> None:
        """Write the question cache to disk. Caller must hold self._cache_lock."""
        try:
            tmp = self._cache_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._cache), encoding="utf-8")
            tmp.replace(self._cache_file)
            self._cache_dirty = 0
        except OSError:
            pass

    def _flush_cache(self) -> None:
        with self._cache_lock:
            self._flush_cache_locked()
