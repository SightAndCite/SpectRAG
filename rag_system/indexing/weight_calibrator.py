from __future__ import annotations
import logging
import statistics
from math import sqrt

import numpy as np

from config import EdgeWeights
from rag_system.indexing.edges.base import RawEdge
from rag_system.indexing.graph_builder import _minmax_normalize

logger = logging.getLogger(__name__)

_ALL_SIGNALS = (
    "semantic", "section", "entity",
    "adjacency", "utility_question", "citation",
)

Pair = tuple[int, int]


def _edge_keys(edges: list[RawEdge]) -> set[Pair]:
    return {(min(e.i, e.j), max(e.i, e.j)) for e in edges}


def calibrate_edge_weights(
    signals: dict[str, list[RawEdge]],
    priors: EdgeWeights,
    fixed: tuple[str, ...] = ("adjacency",),
    floor: float = 0.01,
) -> EdgeWeights:
    """
    Derive per-signal edge weights from how much information each signal carries
    on *this* corpus, instead of using the hardcoded priors (Stage A).

        importance(signal) = sqrt(n_edges) * dispersion
            sqrt(n_edges) — sub-linear coverage: a signal that fires on almost
                            no edges (e.g. citation on non-academic docs) shrinks
                            toward 0; naturally dense signals don't dominate.
            dispersion    — population std of the signal's min-max-normalized
                            scores; ~0 for degenerate/constant signals.

    'fixed' signals keep their prior weight. adjacency is the default: it is
    structurally meaningful (sequential reading order) but its raw score is a
    constant 1.0 -> zero dispersion, so it must not be calibrated to zero. The
    remaining budget (1 - sum of fixed priors) is distributed across the other
    signals in proportion to importance, with a small floor so a live-but-weak
    signal isn't fully silenced.

    Label-free heuristic by design. Swap the body for a self-supervised fit
    (Stage B) without changing any caller.
    """
    prior_map = {name: getattr(priors, name) for name in _ALL_SIGNALS}

    importance: dict[str, float] = {}
    for name in _ALL_SIGNALS:
        if name in fixed:
            continue
        scores = list(_minmax_normalize(signals.get(name, [])).values())
        if len(scores) < 2:
            importance[name] = 0.0
            continue
        importance[name] = sqrt(len(scores)) * statistics.pstdev(scores)

    fixed_total = sum(prior_map[f] for f in fixed)
    budget = max(0.0, 1.0 - fixed_total)
    imp_total = sum(importance.values())

    if imp_total <= 0.0:
        logger.warning(
            "Edge-weight calibration: no informative signals on this corpus — "
            "falling back to hardcoded priors."
        )
        return priors

    weights: dict[str, float] = {f: prior_map[f] for f in fixed}
    for name, imp in importance.items():
        weights[name] = budget * (imp / imp_total)

    # Floor live-but-tiny signals, then renormalize the non-fixed group back to
    # the budget so the full vector still sums to 1.0.
    live = [n for n in importance if importance[n] > 0.0]
    if floor > 0.0 and live:
        for n in live:
            weights[n] = max(weights[n], floor)
        scale = budget / sum(weights[n] for n in live)
        for n in live:
            weights[n] *= scale

    calibrated = EdgeWeights(**{name: weights.get(name, 0.0) for name in _ALL_SIGNALS})

    logger.info(
        "Edge-weight calibration (prior -> calibrated):\n%s",
        "\n".join(
            f"  {name:18s} {prior_map[name]:.3f} -> {getattr(calibrated, name):.3f}"
            f"   (edges={len(signals.get(name, [])):>4d})"
            for name in _ALL_SIGNALS
        ),
    )
    return calibrated


def fit_edge_weights(
    signals: dict[str, list[RawEdge]],
    priors: EdgeWeights,
    n_chunks: int,
    labels: tuple[str, ...] = ("adjacency", "section"),
    features: tuple[str, ...] = ("semantic", "entity", "utility_question", "citation"),
    floor: float = 0.01,
    neg_ratio: float = 1.0,
    min_positives: int = 20,
    seed: int = 0,
) -> EdgeWeights:
    """
    Self-supervised edge weights (Stage B).

    Builds a relatedness target from the corpus itself and fits a logistic
    regression to learn how much each *feature* signal predicts it:

        positives = chunk pairs linked by a 'labels' signal (adjacency / section)
        negatives = random distant pairs not in the positive set
        features  = min-max-normalized scores of the 'features' signals
                    (semantic / entity / utility_question / citation)

    The label signals are deliberately excluded from the features (no leakage):
    they teach the target and keep their prior weight. The learned coefficients
    (clamped to >= 0) are normalized to fill the remaining budget
    (1 - sum of label priors). A held-out AUC is logged so you can tell whether
    the fit actually separates related from unrelated pairs.

    Same return type and caller contract as calibrate_edge_weights, so it slots
    into the pipeline behind the 'fit' mode with no other changes.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    prior_map = {name: getattr(priors, name) for name in _ALL_SIGNALS}

    norm = {f: _minmax_normalize(signals.get(f, [])) for f in features}

    positives: set[Pair] = set()
    for lab in labels:
        positives |= _edge_keys(signals.get(lab, []))

    if len(positives) < min_positives or n_chunks < 4:
        logger.warning(
            "Edge-weight fit: only %d positive pairs (need >= %d) — "
            "falling back to hardcoded priors.",
            len(positives), min_positives,
        )
        return priors

    rng = np.random.default_rng(seed)
    n_neg_target = int(len(positives) * neg_ratio)
    negatives: set[Pair] = set()
    attempts, max_attempts = 0, n_neg_target * 50 + 1000
    while len(negatives) < n_neg_target and attempts < max_attempts:
        i, j = rng.integers(0, n_chunks, size=2)
        attempts += 1
        if i == j:
            continue
        pair = (int(min(i, j)), int(max(i, j)))
        if pair in positives or pair in negatives:
            continue
        negatives.add(pair)

    if not negatives:
        logger.warning("Edge-weight fit: could not sample negatives — using priors.")
        return priors

    def _row(pair: Pair) -> list[float]:
        return [norm[f].get(pair, 0.0) for f in features]

    X = np.array([_row(p) for p in positives] + [_row(p) for p in negatives])
    y = np.array([1] * len(positives) + [0] * len(negatives))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_tr, y_tr)

    try:
        auc = float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
    except ValueError:
        auc = float("nan")

    coef = np.clip(model.coef_.ravel(), 0.0, None)  # negative = signal hurts -> drop
    if coef.sum() <= 0.0:
        logger.warning(
            "Edge-weight fit: no feature has positive predictive weight "
            "(AUC=%.3f) — falling back to priors.", auc,
        )
        return priors

    label_total = sum(prior_map[l] for l in labels)
    budget = max(0.0, 1.0 - label_total)

    weights: dict[str, float] = {l: prior_map[l] for l in labels}
    for f, c in zip(features, coef):
        weights[f] = budget * (c / coef.sum())

    live = [f for f, c in zip(features, coef) if c > 0.0]
    if floor > 0.0 and live:
        for f in live:
            weights[f] = max(weights[f], floor)
        scale = budget / sum(weights[f] for f in live)
        for f in live:
            weights[f] *= scale

    fitted = EdgeWeights(**{name: weights.get(name, 0.0) for name in _ALL_SIGNALS})

    logger.info(
        "Edge-weight fit (held-out AUC=%.3f; prior -> fitted):\n%s",
        auc,
        "\n".join(
            f"  {name:18s} {prior_map[name]:.3f} -> {getattr(fitted, name):.3f}"
            f"   ({'label' if name in labels else 'feature'},"
            f" edges={len(signals.get(name, [])):>4d})"
            for name in _ALL_SIGNALS
        ),
    )
    return fitted
