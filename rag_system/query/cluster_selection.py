from __future__ import annotations
import logging
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from rag_system.models import Chunk
from config import Config

logger = logging.getLogger(__name__)


class ClusterAwareSelector:
    """
    Stage 4 — Greedy context selection (SpectRAG spec).

    Marginal utility per iteration:

      Δ(c_i | C_q) = w_diff · s'_i
                   + w_raw  · sim(e_q, e_i)
                   − w_red  · max_cosine_sim_to_selected
                   + w_con  · graph_connectivity_to_selected
                   + w_div  · cluster_novelty_bonus

    Cluster labels come from SpectralClusterAssigner (offline), falling back
    to query-time K-Means when the index predates offline clustering.
    """

    def __init__(self, cfg: Config) -> None:
        self.mode             = cfg.retrieval.selection_mode
        self.mmr_lambda       = cfg.retrieval.mmr_lambda
        self.protected_core_k = cfg.retrieval.protected_core_k
        self.n_clusters       = cfg.retrieval.n_clusters
        self.final_k          = cfg.retrieval.final_context_k
        # token_budget mode: keep every meaningfully-relevant chunk up to a token cap.
        self.token_budget       = cfg.retrieval.token_budget
        self.token_min_rel      = cfg.retrieval.token_budget_min_rel_ratio
        self.token_tie_band     = cfg.retrieval.token_budget_tie_band
        self.token_head_k       = cfg.retrieval.token_budget_head_k
        self.token_tail_min_rel = cfg.retrieval.token_budget_tail_min_rel_ratio
        self.token_backfill     = cfg.retrieval.token_budget_backfill
        # anchor_expand_ce mode.
        self.anchor_core_k      = cfg.retrieval.anchor_core_k
        self.anchor_neighbors   = cfg.retrieval.anchor_neighbors
        self.anchor_ce_threshold = cfg.retrieval.anchor_ce_threshold
        self._reranker_model = None
        self._tiktoken_encoding = cfg.indexing.tiktoken_encoding
        self._enc = None
        self.w_diff     = cfg.retrieval.sel_diffused_relevance
        self.w_raw      = cfg.retrieval.sel_raw_similarity
        self.w_red      = cfg.retrieval.sel_anti_redundancy
        self.w_con      = cfg.retrieval.sel_connectivity
        self.w_div      = cfg.retrieval.sel_diversity

    def _cluster_at_query_time(
        self, candidate_indices: list[int], chunks: list[Chunk]
    ) -> dict[int, int]:
        """Fallback K-Means over the candidate set for indexes without offline
        labels. Chunks lacking spectral coords each get a unique label so they
        always count as novel (never falsely share a cluster)."""
        valid = [
            (gi, chunks[gi].spectral_coords)
            for gi in candidate_indices
            if chunks[gi].spectral_coords is not None
        ]
        cluster_of: dict[int, int] = {}
        if len(valid) > 1 and valid[0][1].shape[0] > 1:
            matrix = np.stack([c for _gi, c in valid])
            k = min(self.n_clusters, len(valid))
            if k > 1:
                km = KMeans(n_clusters=k, random_state=42, n_init="auto")
                labels = km.fit_predict(matrix)
            else:
                labels = np.zeros(len(valid), dtype=int)
            for (gi, _c), lab in zip(valid, labels):
                cluster_of[gi] = int(lab)
        # Coordless (or trivially few) chunks → unique own-cluster labels.
        next_label = self.n_clusters
        for gi in candidate_indices:
            if gi not in cluster_of:
                cluster_of[gi] = next_label
                next_label += 1
        return cluster_of

    def _mmr_select(
        self,
        candidate_indices: list[int],
        diffused_scores: dict[int, float],
        chunks: list[Chunk],
    ) -> list[int]:
        """Maximal Marginal Relevance over the spectral-diffused relevance.

        Greedily pick the chunk maximizing
            λ·relevance(c) − (1−λ)·max_cosine(c, already_selected),
        so a near-duplicate of an already-chosen chunk is skipped in favour of
        the next distinct relevant chunk. Keeps relevance (high λ) while spending
        redundant slots on additional coverage — more information than a plain
        top-k for the same k. Embeddings are L2-normalized, so dot = cosine.
        """
        rel = np.array([diffused_scores.get(gi, 0.0) for gi in candidate_indices], dtype=np.float64)
        rmax = rel.max() if rel.size else 0.0
        if rmax > 0:
            rel = rel / rmax
        rel_of = {gi: float(rel[i]) for i, gi in enumerate(candidate_indices)}

        lam = self.mmr_lambda
        selected: list[int] = []
        selected_emb: list[np.ndarray] = []
        remaining = list(candidate_indices)

        for _ in range(min(self.final_k, len(candidate_indices))):
            best_gi, best_score = -1, -np.inf
            for gi in remaining:
                if selected_emb:
                    emb = chunks[gi].embedding
                    redundancy = max(float(emb @ e) for e in selected_emb)
                else:
                    redundancy = 0.0
                score = lam * rel_of[gi] - (1.0 - lam) * redundancy
                if score > best_score:
                    best_score, best_gi = score, gi
            if best_gi == -1:
                break
            selected.append(best_gi)
            selected_emb.append(chunks[best_gi].embedding)
            remaining.remove(best_gi)
        return selected

    def _encoder(self):
        if self._enc is None:
            import tiktoken
            self._enc = tiktoken.get_encoding(self._tiktoken_encoding)
        return self._enc

    def _reranker(self):
        if self._reranker_model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # noqa: TRY003
                raise RuntimeError(
                    "anchor_expand_ce needs sentence-transformers: "
                    "`pip install sentence-transformers`"
                ) from exc
            logger.info("Loading cross-encoder ms-marco-MiniLM-L-6-v2 (first use downloads it)…")
            self._reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._reranker_model

    def _anchor_expand_ce_select(
        self,
        candidate_indices: list[int],
        diffused_scores: dict[int, float],
        chunks: list[Chunk],
        graph: nx.Graph,
        question: str | None,
    ) -> list[int]:
        """Graph-anchored, cross-encoder-validated, adaptive selection.

        1. Protected core: top anchor_core_k by diffused score, always kept.
        2. Expansion: walk each core chunk's graph edges (strongest first) to its
           unchosen neighbors — anchor_neighbors each — building an expansion pool.
        3. Cross-encoder ranks the pool against the query.
        4. Add pool chunks that clear the CE threshold, in CE order, as FULL chunks
           until the token budget fills. No summarization, no fixed final-k.
        """
        ranked = sorted(candidate_indices, key=lambda gi: diffused_scores.get(gi, 0.0), reverse=True)
        if not ranked:
            return []
        core = ranked[: min(self.anchor_core_k, len(ranked))]
        chosen: set[int] = set(core)

        # Global-index lookup for candidate chunk-ids (subgraph nodes are candidates).
        idx_of = {chunks[gi].chunk_id: gi for gi in candidate_indices}

        # 2. Graph-anchored expansion — strongest edges first, skip already-chosen.
        expansion: list[int] = []
        for gi in core:
            cid = chunks[gi].chunk_id
            if cid not in graph:
                continue
            neighbors = sorted(
                graph[cid].items(), key=lambda kv: kv[1].get("weight", 0.0), reverse=True
            )
            added = 0
            for nbr_id, _attrs in neighbors:
                nidx = idx_of.get(nbr_id)
                if nidx is None or nidx in chosen:
                    continue
                expansion.append(nidx)
                chosen.add(nidx)
                added += 1
                if added >= self.anchor_neighbors:
                    break

        # 3. Cross-encoder rank the expansion pool (fall back to diffused order).
        if expansion and question:
            scores = self._reranker().predict([(question, chunks[gi].text) for gi in expansion])
            order = [gi for gi, _s in sorted(zip(expansion, scores), key=lambda x: x[1], reverse=True)]
            ce_of = {gi: float(s) for gi, s in zip(expansion, scores)}
        else:
            order = sorted(expansion, key=lambda gi: diffused_scores.get(gi, 0.0), reverse=True)
            ce_of = {}

        # 4. Core (always) + CE-validated neighbors as full chunks up to the budget.
        enc = self._encoder()
        selected = list(core)
        used = sum(len(enc.encode(chunks[gi].text)) for gi in core)
        for gi in order:
            if ce_of and ce_of.get(gi, 0.0) < self.anchor_ce_threshold:
                continue
            n_tok = len(enc.encode(chunks[gi].text))
            if used + n_tok > self.token_budget:
                break
            selected.append(gi)
            used += n_tok
        return selected

    def _coverage_order(
        self,
        ranked: list[int],
        diffused_scores: dict[int, float],
        chunks: list[Chunk],
    ) -> list[int]:
        """Reorder a score-descending list so that, within a small score band,
        chunks from not-yet-covered source documents come first. Only near-ties
        move (band = token_tie_band × the best remaining score), so relevance
        ordering is essentially preserved while multi-document coverage is
        promoted — diversity at ~zero relevance cost."""
        if self.token_tie_band <= 0.0:
            return ranked
        remaining = list(ranked)  # already sorted by score, descending
        covered: set[str] = set()
        order: list[int] = []
        while remaining:
            best = diffused_scores.get(remaining[0], 0.0)
            cutoff = best - self.token_tie_band * best
            band = [gi for gi in remaining if diffused_scores.get(gi, 0.0) >= cutoff]
            # Prefer uncovered documents; break further ties by higher score.
            band.sort(key=lambda gi: (chunks[gi].doc_id in covered, -diffused_scores.get(gi, 0.0)))
            pick = band[0]
            order.append(pick)
            covered.add(chunks[pick].doc_id)
            remaining.remove(pick)
        return order

    def _token_budget_select(
        self,
        candidate_indices: list[int],
        diffused_scores: dict[int, float],
        chunks: list[Chunk],
    ) -> list[int]:
        """Keep relevant chunks until a token budget fills. The top
        token_head_k chunks (answer-bearing core) use the lenient head floor;
        chunks beyond that must clear the stricter tail floor — a precision gate
        that keeps the head which wins most domains while stopping graph-expanded
        but tangential material from diluting the answer. No k-cap; the budget is
        the limit. Within near-tied scores, document-coverage ordering surfaces
        multi-source context at ~zero relevance cost. Best-performing mode in the
        Legal Stage-4 A/B (see generation_eval/legal.py)."""
        ranked = sorted(
            candidate_indices,
            key=lambda gi: diffused_scores.get(gi, 0.0),
            reverse=True,
        )
        if not ranked:
            return []
        top = diffused_scores.get(ranked[0], 0.0)
        head_floor = self.token_min_rel * top
        tail_floor = self.token_tail_min_rel * top
        order = self._coverage_order(ranked, diffused_scores, chunks)
        enc = self._encoder()
        tok_of: dict[int, int] = {}

        def _tok(gi: int) -> int:
            if gi not in tok_of:
                tok_of[gi] = len(enc.encode(chunks[gi].text))
            return tok_of[gi]

        # Pass 1 — precision-gated fill. Head chunks use the lenient floor; once
        # the head is filled the stricter tail floor applies. Sub-floor chunks are
        # remembered (not dropped) so the budget can be backfilled below.
        selected: list[int] = []
        used = 0
        gated_out: list[int] = []
        budget_full = False
        for gi in order:
            floor = head_floor if len(selected) < self.token_head_k else tail_floor
            if diffused_scores.get(gi, 0.0) < floor:
                gated_out.append(gi)
                continue
            n_tok = _tok(gi)
            if selected and used + n_tok > self.token_budget:
                budget_full = True
                break
            selected.append(gi)
            used += n_tok

        # Pass 2 — budget backfill. The tail gate must never leave the token
        # budget under-used: on questions where few candidates clear it, that
        # starved SpectRAG's context below naive's and cost the answer. Fill the
        # remaining space with gated-out chunks by relevance — but ONLY those that
        # still clear the head floor. So backfill = "relax the tail gate when the
        # budget is unused", never "admit genuinely-irrelevant chunks" (which
        # re-dilutes precision). gated_out is sorted descending, so the first
        # sub-head-floor chunk ends the backfill.
        if self.token_backfill and not budget_full:
            for gi in sorted(gated_out, key=lambda g: diffused_scores.get(g, 0.0), reverse=True):
                if diffused_scores.get(gi, 0.0) < head_floor:
                    break
                n_tok = _tok(gi)
                if selected and used + n_tok > self.token_budget:
                    break
                selected.append(gi)
                used += n_tok
        return selected

    def _protected_mmr_select(
        self,
        candidate_indices: list[int],
        diffused_scores: dict[int, float],
        chunks: list[Chunk],
    ) -> list[int]:
        """Protected-core + MMR tail.

        The top `protected_core_k` candidates by spectral-diffused score are kept
        unconditionally — the answer-bearing head is never filtered (100%
        retention of the most-relevant chunks by construction). The remaining
        slots up to `final_k` are filled by MMR over the rest, with redundancy
        measured against everything already selected (core included). So the
        extra budget beyond the core is spent on DISTINCT, graph-discovered
        coverage rather than near-duplicates of the head. Embeddings are
        L2-normalized, so dot = cosine.
        """
        ranked = sorted(
            candidate_indices,
            key=lambda gi: diffused_scores.get(gi, 0.0),
            reverse=True,
        )
        budget = min(self.final_k, len(ranked))
        core_n = min(self.protected_core_k, budget)

        # Protected core: highest diffused scores, kept as-is.
        selected: list[int] = ranked[:core_n]
        selected_emb: list[np.ndarray] = [chunks[gi].embedding for gi in selected]

        remaining = ranked[core_n:]
        if not remaining or len(selected) >= budget:
            return selected

        # Normalized diffused relevance for the MMR tail.
        rel = np.array([diffused_scores.get(gi, 0.0) for gi in remaining], dtype=np.float64)
        rmax = rel.max() if rel.size else 0.0
        if rmax > 0:
            rel = rel / rmax
        rel_of = {gi: float(rel[i]) for i, gi in enumerate(remaining)}

        lam = self.mmr_lambda
        while len(selected) < budget and remaining:
            best_gi, best_score = -1, -np.inf
            for gi in remaining:
                emb = chunks[gi].embedding
                redundancy = max(float(emb @ e) for e in selected_emb) if selected_emb else 0.0
                score = lam * rel_of[gi] - (1.0 - lam) * redundancy
                if score > best_score:
                    best_score, best_gi = score, gi
            if best_gi == -1:
                break
            selected.append(best_gi)
            selected_emb.append(chunks[best_gi].embedding)
            remaining.remove(best_gi)
        return selected

    def select(
        self,
        candidate_indices: list[int],
        diffused_scores: dict[int, float],
        all_scores: np.ndarray,
        chunks: list[Chunk],
        graph: nx.Graph,
        question: str | None = None,
    ) -> list[int]:
        if not candidate_indices:
            return []

        if self.mode == "anchor_expand_ce":
            return self._anchor_expand_ce_select(
                candidate_indices, diffused_scores, chunks, graph, question)

        # Relevance-first replacement: keep the highest spectral-diffused scores.
        # Stages 1–3 already did seed fusion, graph expansion and spectral
        # re-ranking, so this preserves the most relevant chunks instead of
        # trading ~1/3 of them away for diversity (see greedy branch below).
        if self.mode == "topk_diffused":
            ranked = sorted(
                candidate_indices,
                key=lambda gi: diffused_scores.get(gi, 0.0),
                reverse=True,
            )
            return ranked[: self.final_k]

        if self.mode == "token_budget":
            return self._token_budget_select(candidate_indices, diffused_scores, chunks)

        if self.mode == "mmr":
            return self._mmr_select(candidate_indices, diffused_scores, chunks)

        if self.mode == "protected_mmr":
            return self._protected_mmr_select(candidate_indices, diffused_scores, chunks)

        n = len(candidate_indices)

        # Cluster labels for the novelty bonus. Prefer the stable global labels
        # assigned offline by SpectralClusterAssigner; only recompute K-Means at
        # query time for legacy indexes built before offline clustering (label -1).
        if all(chunks[gi].cluster_label >= 0 for gi in candidate_indices):
            cluster_of = {gi: chunks[gi].cluster_label for gi in candidate_indices}
        else:
            cluster_of = self._cluster_at_query_time(candidate_indices, chunks)

        # Normalise diffused relevance
        rel = np.array([diffused_scores.get(gi, 0.0) for gi in candidate_indices])
        rel_max = rel.max()
        if rel_max > 0:
            rel /= rel_max
        rel_of = {candidate_indices[li]: float(rel[li]) for li in range(n)}

        # Normalise raw query similarity
        raw = np.clip([all_scores[gi] for gi in candidate_indices], 0.0, None)
        raw_max = raw.max()
        if raw_max > 0:
            raw /= raw_max
        raw_of = {candidate_indices[li]: float(raw[li]) for li in range(n)}

        # Greedy selection
        selected:             list[int]        = []
        selected_set:         set[int]         = set()
        selected_embeddings:  list[np.ndarray] = []
        covered_clusters:     set[int]         = set()

        for _ in range(min(self.final_k, n)):
            best_gi    = -1
            best_score = -np.inf

            for gi in candidate_indices:
                if gi in selected_set:
                    continue

                relevance = rel_of[gi]
                raw_sim   = raw_of[gi]

                # Anti-redundancy: penalise cosine similarity to already-selected
                if selected_embeddings:
                    emb = chunks[gi].embedding
                    redundancy = max(float(emb @ sel) for sel in selected_embeddings)
                else:
                    redundancy = 0.0

                # Graph connectivity: total edge weight to already-selected chunks
                cid = chunks[gi].chunk_id
                connectivity = 0.0
                if selected and cid in graph:
                    for sel_gi in selected:
                        sel_cid = chunks[sel_gi].chunk_id
                        if graph.has_edge(cid, sel_cid):
                            connectivity += graph[cid][sel_cid].get("weight", 0.0)
                    connectivity = min(1.0, connectivity)

                # Cluster novelty: full bonus for an uncovered topic cluster
                diversity = 0.0 if cluster_of[gi] in covered_clusters else 1.0

                score = (
                    self.w_diff * relevance
                    + self.w_raw  * raw_sim
                    - self.w_red  * redundancy
                    + self.w_con  * connectivity
                    + self.w_div  * diversity
                )

                if score > best_score:
                    best_score = score
                    best_gi    = gi

            if best_gi == -1:
                break

            selected.append(best_gi)
            selected_set.add(best_gi)
            selected_embeddings.append(chunks[best_gi].embedding)
            covered_clusters.add(cluster_of[best_gi])

        return selected
