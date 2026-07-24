from __future__ import annotations
import numpy as np
from rag_system.models import Chunk
from config import Config


class RelevanceDiffuser:
    """
    Stage 3 — Spectral proximity re-scoring (SpectRAG spec Step 7).

    Re-scores each candidate by how close it is to the seed set in
    Laplacian eigenspace (spectral coordinates), then blends with the
    original cosine similarity:

        combined(c) = (1 − α) · cosine_sim(q, c)
                    +      α  · max_s∈S₀ cosine(z_c, z_s)

    No graph operations at query time — uses only the spectral_coords
    stored offline during indexing.
    """

    def __init__(self, cfg: Config) -> None:
        self._alpha = cfg.retrieval.diffusion_spectral_alpha
        self._agg   = cfg.retrieval.diffusion_seed_aggregation
        self._topk  = cfg.retrieval.diffusion_seed_topk
        self._mode  = cfg.retrieval.diffusion_mode
        self._ppr_damping = cfg.retrieval.ppr_damping

    def _spectral_proximity(
        self,
        candidate_indices: list[int],
        seed_indices: list[int],
        all_scores: np.ndarray,
        chunks: list[Chunk],
    ) -> dict[int, float] | None:
        """Per-candidate cosine proximity to the seeds in Laplacian eigenspace,
        aggregated across seeds. None if no seed has spectral coords."""
        kept = [
            (i, chunks[i].spectral_coords) for i in seed_indices
            if chunks[i].spectral_coords is not None
        ]
        if not kept:
            return None

        S = np.stack([c for _i, c in kept]).astype(np.float64)  # (n_seeds, dim)
        norms = np.linalg.norm(S, axis=1, keepdims=True)
        S = S / np.where(norms > 0, norms, 1.0)                # L2-normalised rows

        seed_w = np.array([max(0.0, float(all_scores[i])) for i, _c in kept], dtype=np.float64)
        if seed_w.sum() <= 0:
            seed_w = np.ones(len(kept), dtype=np.float64)
        seed_w = seed_w / seed_w.sum()
        topk = min(self._topk, S.shape[0])

        out: dict[int, float] = {}
        for gi in candidate_indices:
            coords = chunks[gi].spectral_coords
            if coords is None:
                out[gi] = 0.0
                continue
            z = coords.astype(np.float64)
            z_norm = np.linalg.norm(z)
            z = z / z_norm if z_norm > 0 else z
            sims = S @ z                                        # (n_seeds,) cosine to each seed
            if self._agg == "max":
                out[gi] = float(sims.max())
            elif self._agg == "relevance_weighted":
                out[gi] = float(np.dot(seed_w, sims))
            else:  # "topk_mean" — mean of the top-k proximities (smoothed max)
                out[gi] = float(sims.mean()) if sims.shape[0] <= topk \
                    else float(np.partition(sims, -topk)[-topk:].mean())
        return out

    def _ppr_proximity(
        self,
        candidate_indices: list[int],
        seed_indices: list[int],
        chunks: list[Chunk],
        graph,
    ) -> dict[int, float] | None:
        """Personalized PageRank from the seeds over the real weighted candidate
        graph, normalised to [0,1]. None if the graph is unusable (no edges, no
        seeds present, or non-convergence)."""
        if graph is None or graph.number_of_edges() == 0:
            return None
        import networkx as nx

        seed_ids = {chunks[i].chunk_id for i in seed_indices}
        present = [sid for sid in seed_ids if sid in graph]
        if not present:
            return None
        personalization = {sid: 1.0 / len(present) for sid in present}
        try:
            pr = nx.pagerank(
                graph, alpha=self._ppr_damping,
                personalization=personalization, weight="weight", max_iter=200,
            )
        except Exception:  # noqa: BLE001 — non-convergence etc. → fall back
            return None
        pmax = max(pr.values()) if pr else 0.0
        if pmax <= 0:
            return None
        return {gi: pr.get(chunks[gi].chunk_id, 0.0) / pmax for gi in candidate_indices}

    def diffuse(
        self,
        candidate_indices: list[int],
        seed_indices: list[int],
        all_scores: np.ndarray,
        chunks: list[Chunk],
        graph=None,
    ) -> dict[int, float]:
        """Returns {chunk_global_idx: combined_score} for every candidate.

        combined = (1 − α)·cosine + α·proximity, where proximity is spectral,
        PPR, or their mean depending on diffusion_mode. Each proximity source
        degrades gracefully to the other (or to pure cosine) when unavailable.
        """
        if not candidate_indices:
            return {}

        spectral = ppr = None
        if self._mode in ("spectral", "hybrid"):
            spectral = self._spectral_proximity(candidate_indices, seed_indices, all_scores, chunks)
        if self._mode in ("ppr", "hybrid"):
            ppr = self._ppr_proximity(candidate_indices, seed_indices, chunks, graph)

        # If nothing usable, fall back to raw cosine ranking.
        if spectral is None and ppr is None:
            return {gi: float(all_scores[gi]) for gi in candidate_indices}

        scores: dict[int, float] = {}
        for gi in candidate_indices:
            if spectral is not None and ppr is not None:
                prox = 0.5 * spectral.get(gi, 0.0) + 0.5 * ppr.get(gi, 0.0)
            elif spectral is not None:
                prox = spectral.get(gi, 0.0)
            else:
                prox = ppr.get(gi, 0.0)
            scores[gi] = (1.0 - self._alpha) * float(all_scores[gi]) + self._alpha * prox
        return scores
