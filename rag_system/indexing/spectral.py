from __future__ import annotations
import logging
import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from rag_system.models import Chunk
from rag_system.indexing.edges.base import RawEdge
from rag_system.indexing.graph_builder import _minmax_normalize
from config import IndexingConfig, EdgeWeights

logger = logging.getLogger(__name__)

_LAYER_ORDER = (
    "semantic", "section", "entity",
    "adjacency", "utility_question", "citation",
)

# Small negative shift below the (non-negative) normalized-Laplacian spectrum.
# Shift-invert around it (L - σI is then positive-definite, so it factorizes)
# targets the smallest eigenvalues far more reliably/quickly than which="SM".
_ARPACK_SHIFT = -1e-3


class SpectralComputer:
    """
    Computes top-k eigenvectors of the normalised graph Laplacian.
    Each chunk's spectral coordinates capture its position in the graph's
    community structure and are reused at query time for clustering and diffusion.

    For graphs above cfg.spectral_large_graph_threshold nodes the solver switches
    from ARPACK to LOBPCG to avoid memory pressure.
    """

    def __init__(self, cfg: IndexingConfig) -> None:
        self._k               = cfg.n_spectral_components
        self._large_threshold = cfg.spectral_large_graph_threshold
        self._lobpcg_max_iter = cfg.spectral_lobpcg_max_iter
        self._lobpcg_tol      = cfg.spectral_lobpcg_tol

    def compute(self, G: nx.Graph, chunks: list[Chunk]) -> np.ndarray:
        """Assigns spectral_coords in-place to each chunk. Returns (n, k) float32."""
        n = len(chunks)
        chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(chunks)}
        nodes = list(G.nodes())
        node_to_row = {nid: i for i, nid in enumerate(nodes)}

        rows, cols, data = [], [], []
        for u, v, attr in G.edges(data=True):
            w = attr.get("weight", 1.0)
            r, c = node_to_row[u], node_to_row[v]
            rows += [r, c]
            cols += [c, r]
            data += [w, w]

        A = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
        L = sp.csgraph.laplacian(A, normed=True)

        vecs = self._solve(L, n)

        result = np.zeros((n, vecs.shape[1]), dtype=np.float32)
        for graph_idx, node_id in enumerate(nodes):
            chunk_idx = chunk_id_to_idx.get(node_id)
            if chunk_idx is not None:
                result[chunk_idx] = vecs[graph_idx].astype(np.float32)

        # Chunks not in the graph stay all-zeros, which collapses them to one
        # point in the visualisation.  Assign them position-interpolated coords
        # within the range of the valid spectral coordinates.
        valid_mask = np.any(result != 0.0, axis=1)
        self._interpolate_isolated(result, chunks, valid_mask)

        for i, chunk in enumerate(chunks):
            chunk.spectral_coords = result[i]

        return result

    def compute_multiplex(
        self,
        signal_edges: dict[str, list[RawEdge]],
        weights: EdgeWeights,
        chunks: list[Chunk],
    ) -> np.ndarray:
        """
        Option 4: spectral coords from an aggregated multi-layer Laplacian.

        Instead of collapsing the six signals into one weighted graph and then
        taking its Laplacian, build a per-signal normalized Laplacian Lₖ and
        aggregate:  L = Σ βₖ·Lₖ, with βₖ = the chosen edge weight for signal k.
        Each layer is degree-normalized on its own, so a sparse but precise
        signal keeps its footing instead of being drowned by a dense one.

        Edges use chunk indices directly (RawEdge.i/j index into `chunks`), so no
        node-id remapping is needed. Assigns spectral_coords in-place.
        """
        n = len(chunks)
        L_multi: sp.spmatrix | None = None
        connected = np.zeros(n, dtype=bool)
        used: list[str] = []

        for name in _LAYER_ORDER:
            beta = getattr(weights, name)
            edges = signal_edges.get(name, [])
            if beta <= 0.0 or not edges:
                continue

            rows, cols, data = [], [], []
            for (i, j), score in _minmax_normalize(edges).items():
                rows += [i, j]
                cols += [j, i]
                data += [score, score]
                connected[i] = True
                connected[j] = True

            A_k = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
            L_k = beta * sp.csgraph.laplacian(A_k, normed=True)
            L_multi = L_k if L_multi is None else L_multi + L_k
            used.append(name)

        if L_multi is None:
            raise ValueError(
                "Multiplex spectral: no usable layers (all weights 0 or empty)."
            )

        logger.info("Multiplex spectral over %d layers: %s", len(used), ", ".join(used))
        vecs = self._solve(sp.csr_matrix(L_multi), n)

        result = vecs.astype(np.float32)
        self._interpolate_isolated(result, chunks, connected)

        for i, chunk in enumerate(chunks):
            chunk.spectral_coords = result[i]

        return result

    def _solve(self, L: sp.spmatrix, n: int) -> np.ndarray:
        """Dispatch to ARPACK or LOBPCG and return the smallest non-trivial eigenvectors."""
        k_req = min(self._k + 1, n - 1)
        # Too few nodes for a meaningful eigenbasis: k_req < 2 means we'd drop the
        # trivial vector and be left with 0 columns (or eigsh(k=0) would raise).
        # Return a valid deterministic 1-D fallback so downstream clustering and
        # diffusion still work on tiny corpora.
        if k_req < 2:
            logger.info("Spectral: graph too small (n=%d) — using 1-D position fallback", n)
            return np.linspace(0.1, 1.0, num=n, dtype=np.float32).reshape(-1, 1)
        logger.info(
            "Spectral: computing %d eigenvectors of %dx%d Laplacian", k_req, n, n
        )
        return (
            self._lobpcg(L, k_req)
            if n > self._large_threshold
            else self._arpack(L, k_req)
        )

    def _interpolate_isolated(
        self, result: np.ndarray, chunks: list[Chunk], valid_mask: np.ndarray
    ) -> None:
        """Give disconnected chunks position-interpolated coords inside the valid range."""
        n = len(chunks)
        n_valid = int(valid_mask.sum())
        if not (0 < n_valid < n):
            return
        v_min = result[valid_mask].min(axis=0)
        v_max = result[valid_mask].max(axis=0)
        v_range = np.where((v_max - v_min) > 0, v_max - v_min, 1.0)
        for i, chunk in enumerate(chunks):
            if not valid_mask[i]:
                t = chunk.position / max(n - 1, 1)
                result[i] = v_min + t * v_range

    def _arpack(self, L: sp.spmatrix, k: int) -> np.ndarray:
        # Deterministic start vector so degenerate spectra give reproducible
        # eigenvectors across identical runs (ARPACK is otherwise randomly seeded).
        v0 = np.ones(L.shape[0], dtype=np.float64) / np.sqrt(L.shape[0])
        try:
            # Preferred: shift-invert around a small negative shift — robust & fast
            # for the smallest eigenvalues.
            eigenvalues, eigenvectors = spla.eigsh(
                L, k=k, sigma=_ARPACK_SHIFT, which="LM", v0=v0
            )
        except Exception as exc:  # noqa: BLE001 — singular factorization, non-convergence…
            logger.warning("Shift-invert eigsh failed (%s); retrying which='SM'", exc)
            try:
                eigenvalues, eigenvectors = spla.eigsh(L, k=k, which="SM", v0=v0)
            except spla.ArpackNoConvergence as exc2:
                logger.warning("ARPACK did not converge, using partial result: %s", exc2)
                eigenvalues, eigenvectors = exc2.eigenvalues, exc2.eigenvectors
        order = np.argsort(eigenvalues)
        return eigenvectors[:, order[1:]]  # drop trivial constant eigenvector

    def _lobpcg(self, L: sp.spmatrix, k: int) -> np.ndarray:
        n = L.shape[0]
        rng = np.random.default_rng(42)
        X0 = rng.standard_normal((n, k))
        X0, _ = np.linalg.qr(X0)
        try:
            eigenvalues, eigenvectors = spla.lobpcg(
                L, X0,
                largest=False,
                maxiter=self._lobpcg_max_iter,
                tol=self._lobpcg_tol,
            )
            order = np.argsort(eigenvalues)
            return eigenvectors[:, order[1:]]
        except Exception as exc:
            logger.warning("LOBPCG failed, falling back to ARPACK: %s", exc)
            return self._arpack(L, k)
