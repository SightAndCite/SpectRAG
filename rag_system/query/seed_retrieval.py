from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import faiss
from rag_system.models import Chunk
from rag_system.indexing.embedder import OllamaEmbedder
from config import Config

from rag_system.store.lexical_index import LexicalIndex, tokenize as _tokenize
from rag_system.store.question_index import QuestionIndex


@dataclass(frozen=True)
class QueryContext:
    """Per-query state needed to score candidates after graph expansion.

    Replaces the (N,) relevance vector that Stage 1 used to build for every
    request. Both maxima are the true corpus-wide values — `dense_max` is the top
    exhaustive inner product, `bm25_max` comes from the single lexical pass — so
    deferring the scoring changes nothing about the result.
    """
    q_emb: np.ndarray                  # (1, D), L2-normalized
    dense_max: float                   # corpus max cosine, for the hybrid blend
    bm25_scores: np.ndarray | None     # (N,) BM25, computed once per request
    bm25_max: float
    hybrid_weight: float               # 0.0 when the lexical blend does not apply


class SeedRetriever:
    """Stage 1 — produce initial seed chunk indices.

    Dense FAISS cosine, optionally augmented with utility-question matching:
    each chunk carries LLM-generated "what question does this answer?" strings
    (built at index time for the utility-question edges). Those are query-shaped
    surrogates of the chunk, so matching the user query against them
    (question↔question) bridges the abstractive-query ↔ passage gap that plain
    dense retrieval struggles with. The dense ranking and the utility-question
    ranking are fused with Reciprocal Rank Fusion (rank-based ⇒ no score-scale
    tuning). No query-time LLM cost — the questions already exist on the chunks.
    """

    def __init__(self, cfg: Config, embedder: OllamaEmbedder) -> None:
        rc = cfg.retrieval
        self.k = rc.seed_k
        self.embedder = embedder
        self.uq_enabled = rc.uq_seed_enabled
        self.uq_k = rc.uq_seed_k
        self.rrf_k = rc.rrf_k
        self.bm25_enabled = rc.bm25_seed_enabled
        self.bm25_k = rc.bm25_seed_k
        self.hybrid_lex_w = rc.hybrid_lexical_weight
        # Utility-question FAISS index, rebuilt when the corpus changes.
        self._uq_index: QuestionIndex | None = None
        self._uq_key: tuple | None = None
        # BM25 lexical index, rebuilt when the corpus changes.
        self._bm25 = None
        self._bm25_key: tuple | None = None

    @staticmethod
    def _corpus_key(chunks: list[Chunk]) -> tuple:
        """Content-based cache key for the corpus. Keyed on size + boundary chunk
        ids (unique per corpus) rather than id(chunks): Python reuses an object's
        id() after it is freed, so a reallocated-but-different list could otherwise
        be mistaken for the same corpus and serve a stale index (see CASE-10)."""
        if not chunks:
            return (0,)
        return (len(chunks), chunks[0].chunk_id, chunks[-1].chunk_id)

    def _rrf(self, ranked_lists: list[list[int]]) -> list[int]:
        """Fuse ranked index lists by Reciprocal Rank Fusion (higher = better)."""
        scores: dict[int, float] = {}
        for lst in ranked_lists:
            for rank, idx in enumerate(lst):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        return [i for i, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    def _question_index(self, chunks: list[Chunk],
                        provided: QuestionIndex | None) -> QuestionIndex | None:
        """The persisted question index, or one built in memory as a fallback.

        Serving passes the artifact published at build time, so no request embeds
        anything. The fallback covers the CLI and indexes built before F13; it
        assigns the built index before publishing its key, so a concurrent caller
        can never observe the key without the artifact and silently drop the
        signal — the shape of bug F11 describes.
        """
        if provided is not None:
            return provided
        key = self._corpus_key(chunks)
        if self._uq_key == key:
            return self._uq_index
        built = QuestionIndex.build(chunks, self.embedder, prefix_role="query")
        self._uq_index = built
        self._uq_key = key            # published only after the artifact exists
        return built

    def _uq_seed_ranking(self, q_emb: np.ndarray, chunks: list[Chunk],
                         provided: QuestionIndex | None) -> list[int]:
        """Chunk indices ranked by best matching utility question (dedup, best rank)."""
        qi = self._question_index(chunks, provided)
        if qi is None or len(qi) == 0:
            return []
        n = min(self.uq_k * 4, len(qi))          # over-fetch; dedup to chunks
        seen: set[int] = set()
        ranked: list[int] = []
        for row in qi.search(q_emb, n)[0]:
            if row < 0:
                continue
            ci = int(qi.to_chunk[row])
            if ci not in seen:
                seen.add(ci)
                ranked.append(ci)
            if len(ranked) >= self.uq_k:
                break
        return ranked

    def _lexical_index(self, chunks: list[Chunk],
                       provided: LexicalIndex | None) -> LexicalIndex | None:
        """The persisted inverted index, or one built in memory as a fallback.

        Serving passes the index published at build time, so no corpus work
        happens on a request. The CLI and tests may not, in which case one is
        built once and cached — still an inverted index rather than rank-bm25,
        which held the tokenised corpus plus a frequency dict per document
        (~15 kB per chunk, about 15 GB at 1M).
        """
        if provided is not None:
            return provided
        key = self._corpus_key(chunks)
        if self._bm25_key == key and self._bm25 is not None:
            return self._bm25
        built = LexicalIndex.build([c.text for c in chunks])
        self._bm25, self._bm25_key = built, key
        return built

    @staticmethod
    def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
        """Top-k indices by descending score, ties broken by ascending index.

        argpartition finds the k largest in O(N) and only those k are sorted, so
        no corpus-wide O(N log N) sort happens. Ties resolve on index, which makes
        the ranking deterministic — np.argsort's default quicksort is unstable, so
        the previous ordering of equal scores was arbitrary.
        """
        n = scores.shape[0]
        if k >= n:
            part = np.arange(n)
        else:
            part = np.argpartition(-scores, k)[:k]
        return part[np.lexsort((part, -scores[part]))]

    def _bm25_ranking(self, scores: np.ndarray) -> list[int]:
        """Chunk indices ranked by BM25 lexical score (positive scores only)."""
        return [int(i) for i in self._top_k(scores, self.bm25_k) if scores[i] > 0.0]

    def retrieve(
        self,
        query: str,
        chunks: list[Chunk],
        faiss_index: faiss.Index,
        lexical: LexicalIndex | None = None,
        questions: QuestionIndex | None = None,
    ) -> tuple[list[int], QueryContext]:
        """
        Returns:
            seed_indices — top-K chunk indices (dense, or RRF-fused with UQ/BM25)
            ctx          — what `score()` needs to rank candidates later

        No corpus-wide relevance vector is built. Stage 3 and Stage 4 only ever
        read scores at seed and candidate positions — a few hundred of them — so
        scoring is deferred until the candidate set is known (see `score`).
        """
        q_emb = self.embedder.embed([query], kind="query")    # (1, D) normalized
        k = min(self.k, len(chunks))

        # Dense ranking (over-fetch a little to give fusion room to work).
        dense_k = min(max(self.k, self.uq_k), len(chunks))
        dense_scores, dense_idx = faiss_index.search(q_emb, dense_k)
        dense_rank = [int(i) for i in dense_idx[0] if i >= 0]

        # The top inner product IS the corpus-wide maximum cosine, because the
        # index is exhaustive and embeddings are L2-normalized. That keeps the
        # hybrid normalizer exact without materializing all N scores.
        dense_max = float(dense_scores[0][0]) if dense_rank else 0.0

        # Hybrid lexical relevance: normalized BM25 is folded into the relevance
        # score so lexical precision reaches Stage-3 diffusion and Stage-4
        # selection, not just the seed choice.
        bm25_full: np.ndarray | None = None
        bm25_max = 0.0
        bm25_rank: list[int] = []
        lex = self._lexical_index(chunks, lexical) if self.bm25_enabled else None
        if lex is not None:
            # One pass over the postings of the query's terms, reused for the seed
            # ranking, the exact corpus maximum, and candidate scoring.
            bm25_full = lex.scores(_tokenize(query))
            bm25_rank = self._bm25_ranking(bm25_full)
            bm25_max = float(np.clip(bm25_full, 0.0, None).max()) if bm25_full.size else 0.0

        ctx = QueryContext(
            q_emb=q_emb,
            dense_max=dense_max,
            bm25_scores=bm25_full,
            bm25_max=bm25_max,
            # A zero maximum means the query matched nothing lexically; the old
            # full-vector path returned None there and skipped the blend entirely.
            hybrid_weight=self.hybrid_lex_w if (bm25_full is not None and bm25_max > 0) else 0.0,
        )

        # Fuse dense with the utility-question and BM25 rankings (whichever are
        # enabled and non-empty) via Reciprocal Rank Fusion.
        ranked_lists = [dense_rank]
        if self.uq_enabled:
            uq_rank = self._uq_seed_ranking(q_emb, chunks, questions)
            if uq_rank:
                ranked_lists.append(uq_rank)
        if bm25_rank:
            ranked_lists.append(bm25_rank)

        if len(ranked_lists) == 1:
            return dense_rank[:k], ctx
        return self._rrf(ranked_lists)[:k], ctx

    def score(
        self,
        ctx: QueryContext,
        chunks: list[Chunk],
        indices: Iterable[int],
    ) -> dict[int, float]:
        """Relevance for the given chunk indices only — Stage 3 and Stage 4 input.

        Same arithmetic as the old corpus-wide vector, evaluated on the candidate
        union instead of all N. Both normalizers are the true corpus maxima
        carried on `ctx`, so results are unchanged.
        """
        idx = sorted({int(i) for i in indices})
        if not idx:
            return {}

        embs = np.stack([chunks[i].embedding for i in idx]).astype(np.float32)
        dense = (embs @ ctx.q_emb.T).ravel().astype(np.float64)

        if ctx.hybrid_weight <= 0.0:
            return {i: float(s) for i, s in zip(idx, dense)}

        lex = np.clip(ctx.bm25_scores[idx], 0.0, None) / ctx.bm25_max
        dense_n = dense / ctx.dense_max if ctx.dense_max > 0 else dense
        blended = (1.0 - ctx.hybrid_weight) * dense_n + ctx.hybrid_weight * lex
        return {i: float(s) for i, s in zip(idx, blended)}
