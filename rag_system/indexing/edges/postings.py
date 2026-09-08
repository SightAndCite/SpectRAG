"""Bounded chunk-pair construction from shared-key posting lists.

Entity/concept edges and citation edges are the same computation: chunks that
share a key get linked. Both previously enumerated *every* pair in every posting
list with no cap, which is quadratic in document frequency.

Document frequency grows with corpus size, so pair count grows quadratically. A
term sitting in 20% of the corpus produces ~3.9K pairs at 453 chunks and ~2e10
at 1M — from one term, materialised into a Python dict before any thresholding
runs. That is the memory blocker in a million-chunk build.

Three bounds fix it, in the order they apply:

  1. Drop the high-document-frequency tail. A key in a large fraction of the
     corpus has near-zero IDF and links everything to everything; it is a
     stopword by behaviour. This is what bounds pairs per key.
  2. Cap how many keys one chunk contributes, keeping its rarest (most
     specific) ones.
  3. Cap how many neighbours one chunk keeps.

Shared keys are then weighted by IDF rather than a flat count, so sharing a rare
key counts for more than sharing a common one.

Setting max_df_ratio=1.0, max_per_chunk=0, max_neighbors=0 and
idf_weighting=False reproduces the previous unbounded behaviour exactly, which
is what lets F12 ablate against today's graph.
"""
from __future__ import annotations

import logging
import heapq
import math
from collections import defaultdict

from rag_system.indexing.edges.base import RawEdge

logger = logging.getLogger(__name__)


def edges_from_postings(
    postings: dict[str, list[int]],
    n_chunks: int,
    *,
    min_df: int = 2,
    max_df_ratio: float = 1.0,
    max_df_abs: int = 0,
    df_floor: int = 0,
    max_per_chunk: int = 0,
    max_neighbors: int = 0,
    idf_weighting: bool = False,
    label: str = "shared-key",
) -> list[RawEdge]:
    """Chunk-pair edges from key -> chunk-index postings, with bounded fan-out.

    `min_df` is the smallest posting length that can form a pair. `max_df_ratio`
    is a fraction of `n_chunks`; keys above it are dropped. `max_per_chunk` and
    `max_neighbors` are 0 for unlimited. Scores are max-normalised to [0, 1].
    """
    if n_chunks <= 1 or not postings:
        return []

    # 1. Deduplicate. NER can yield the same normalised term twice for one chunk,
    #    which previously inflated that term's df and could pair a chunk with
    #    itself. Sorted for deterministic pair ordering.
    deduped: dict[str, list[int]] = {k: sorted(set(v)) for k, v in postings.items()}

    # Pairs per key go as df², so the ceiling that bounds WORK must be absolute:
    # a ratio scales with the corpus and therefore does not bound anything. The
    # ratio expresses the IDF argument (a key in a large fraction of the corpus
    # is not discriminative), the floor stops a small corpus being over-pruned
    # where there is no work problem to solve, and the absolute caps the cost.
    if max_df_ratio >= 1.0 and not max_df_abs:
        df_ceiling = n_chunks
    else:
        ratio_cap = int(n_chunks * max_df_ratio) if max_df_ratio < 1.0 else n_chunks
        df_ceiling = max(ratio_cap, df_floor)
        if max_df_abs:
            df_ceiling = min(df_ceiling, max_df_abs)
        df_ceiling = max(df_ceiling, min_df)
    kept: dict[str, list[int]] = {}
    dropped_hubs = 0
    for key, idxs in deduped.items():
        df = len(idxs)
        if df < min_df:
            continue
        if df > df_ceiling:
            dropped_hubs += 1
            continue
        kept[key] = idxs

    if not kept:
        if dropped_hubs:
            logger.info("%s: all %d keys exceeded the df ceiling (%d)",
                        label, dropped_hubs, df_ceiling)
        return []

    # True corpus document frequency, captured before the per-chunk cap trims
    # postings — IDF should reflect the corpus, not what survived truncation.
    true_df = {k: len(v) for k, v in kept.items()}

    # 2. Per-chunk fan-out cap: keep each chunk's rarest keys, which carry the
    #    most information. A bibliography page emits hundreds of references and
    #    would otherwise link to every other bibliography page.
    if max_per_chunk > 0:
        by_chunk: dict[int, list[str]] = defaultdict(list)
        for key, idxs in kept.items():
            for i in idxs:
                by_chunk[i].append(key)
        allowed: dict[int, set[str]] = {}
        for i, keys in by_chunk.items():
            if len(keys) > max_per_chunk:
                keys = sorted(keys, key=lambda k: (true_df[k], k))[:max_per_chunk]
            allowed[i] = set(keys)
        trimmed: dict[str, list[int]] = {}
        for key, idxs in kept.items():
            survivors = [i for i in idxs if key in allowed.get(i, ())]
            if len(survivors) >= min_df:
                trimmed[key] = survivors
        kept = trimmed

    # 3+4. Accumulate and cap in ONE pass, chunk by chunk.
    #
    # The degree cap must be applied DURING accumulation, not after. A ceiling on
    # document frequency bounds pairs per key, but not their sum: at a ceiling of
    # 200, every 200 postings can contribute another 19,900 pairs, so a corpus
    # with millions of postings still builds a pair dict far too large to hold.
    # Pruning afterwards cannot prevent that — the dict is the thing that OOMs.
    #
    # Instead, gather one chunk's neighbours at a time and keep only its top-K
    # before moving on. Transient memory is then bounded by
    # max_per_chunk × df_ceiling (the most neighbours one chunk can reach), and
    # the result by n_chunks × max_neighbors. Neither term is quadratic.
    by_chunk: dict[int, list[str]] = defaultdict(list)
    for key, idxs in kept.items():
        for i in idxs:
            by_chunk[i].append(key)

    weight = {
        k: (math.log(n_chunks / true_df[k]) if idf_weighting else 1.0)
        for k in kept
    }

    scores: dict[tuple[int, int], float] = {}
    for i, keys in by_chunk.items():
        local: dict[int, float] = {}
        for key in keys:
            w = weight[key]
            if w <= 0.0:
                continue
            for j in kept[key]:
                if j != i:
                    local[j] = local.get(j, 0.0) + w
        if not local:
            continue
        if max_neighbors > 0 and len(local) > max_neighbors:
            # Strongest first, ties by lowest index so the result is deterministic.
            items = heapq.nlargest(max_neighbors, local.items(),
                                   key=lambda kv: (kv[1], -kv[0]))
        else:
            items = local.items()
        # An edge survives if EITHER endpoint keeps it, so a rare-but-real link is
        # not lost because the other end happens to be popular. The score is the
        # same computed from either side, so writing twice is consistent.
        for j, s in items:
            scores[(i, j) if i < j else (j, i)] = s

    if not scores:
        return []

    max_score = max(scores.values())
    edges = [RawEdge(i, j, s / max_score) for (i, j), s in scores.items()]
    logger.info(
        "%s: %d keys kept (%d dropped above df ceiling %d), %d edges",
        label, len(kept), dropped_hubs, df_ceiling, len(edges),
    )
    return edges
