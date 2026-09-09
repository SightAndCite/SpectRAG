"""Vector index construction, shared by the chunk and question indexes.

`IndexFlatIP` scans every vector on every search, so single-query latency grows
linearly with the corpus: measured at 2.1 ms for 50K, 20.8 ms for 500K, and
therefore roughly 42 ms at 1M. That is about 24 queries/second of total capacity
for vector search alone, against a 20 RPS target that must also fit graph
expansion, diffusion, selection and generation. HNSW is flat in corpus size —
0.27 ms at 500K, the same as at 50K.

The trade is real, unlike the rest of this series: HNSW is approximate. Recall
depends heavily on vector geometry, and the difference is not subtle — at 200K
vectors, recall@20 was 1.000 on clustered data and 0.361 on isotropic data.
Trained embeddings are clustered, so the first figure is the relevant one, but
that must be measured on the actual encoder rather than assumed.

DEFAULT IS THEREFORE "flat". The mechanism is here and persisted, but switching
it on is a measured decision for F12, not a default set from a synthetic
embedder. Flipping `vector_index_type` is the whole change.

Deletion: HNSW cannot remove vectors. F16's tombstones and compaction will need
either query-time filtering of deleted ids plus periodic rebuild, or an index
type that supports removal. Recorded here so that design starts from it.
"""
from __future__ import annotations

import logging

import faiss
import numpy as np

logger = logging.getLogger(__name__)

FLAT = "flat"
HNSW = "hnsw"


def build_vector_index(
    vectors: np.ndarray,
    *,
    index_type: str = FLAT,
    m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 128,
) -> faiss.Index:
    """Build an inner-product index over L2-normalised vectors."""
    if vectors.ndim != 2:
        raise ValueError(f"expected a 2-D vector array, got shape {vectors.shape}")
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    d = vectors.shape[1]

    if index_type == FLAT:
        index: faiss.Index = faiss.IndexFlatIP(d)
    elif index_type == HNSW:
        index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        logger.info("Building HNSW index: %d vectors, M=%d, efConstruction=%d "
                    "(approximate — recall depends on vector geometry)",
                    len(vectors), m, ef_construction)
    else:
        raise ValueError(
            f"Unknown vector_index_type {index_type!r}; expected {FLAT!r} or {HNSW!r}")

    index.add(vectors)
    return index


def apply_search_params(index: faiss.Index, ef_search: int) -> faiss.Index:
    """Set query-time parameters on a loaded index.

    `efSearch` is persisted inside the index file, so an index built under one
    setting would keep using it. Applying the configured value on load makes the
    recall/latency trade tunable without a rebuild.
    """
    hnsw = getattr(index, "hnsw", None)
    if hnsw is not None and ef_search > 0:
        hnsw.efSearch = ef_search
    return index


def describe(index: faiss.Index) -> str:
    """Short label for logs and manifests."""
    hnsw = getattr(index, "hnsw", None)
    if hnsw is not None:
        return f"hnsw(efSearch={hnsw.efSearch})"
    return "flat"
