from __future__ import annotations
import logging
import numpy as np
from sklearn.cluster import KMeans
from rag_system.models import Chunk
from config import IndexingConfig

logger = logging.getLogger(__name__)


class SpectralClusterAssigner:
    """
    Offline spectral clustering (SpectRAG spec — indexing phase).

    Runs K-Means on all chunks' spectral coordinates and stores the resulting
    cluster label directly on each Chunk. Query time uses these labels for the
    cluster-novelty bonus without re-running K-Means on every request.
    """

    def __init__(self, cfg: IndexingConfig) -> None:
        self._n_clusters = cfg.n_offline_clusters

    def assign(self, chunks: list[Chunk]) -> None:
        """Assigns cluster_label in-place to every chunk that has spectral_coords."""
        valid_idx = [i for i, c in enumerate(chunks) if c.spectral_coords is not None]
        if not valid_idx:
            logger.warning("No spectral coords found — cluster labels not assigned.")
            return

        # Fewer than two clusterable chunks: K-Means is undefined — put them all in
        # cluster 0 (a single chunk can't crash the whole index).
        n = len(valid_idx)
        if n < 2:
            for chunk_idx in valid_idx:
                chunks[chunk_idx].cluster_label = 0
            return

        matrix = np.stack([chunks[i].spectral_coords for i in valid_idx])
        k = max(2, min(self._n_clusters, n))

        logger.info("Spectral clustering: K=%d over %d chunks", k, n)
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(matrix)

        for pos, chunk_idx in enumerate(valid_idx):
            chunks[chunk_idx].cluster_label = int(labels[pos])
