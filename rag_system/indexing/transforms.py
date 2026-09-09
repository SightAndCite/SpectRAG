"""Frozen per-signal score transforms.

`GraphBuilder` normalises each signal's raw scores against the corpus-wide min
and max of that signal. Recomputing those extrema on every build means a single
new edge stronger than anything before it rescales every edge already published:
measured, base edges with identical raw scores normalise to 0.500 and 1.000
alone but 0.308 and 0.615 once one stronger edge arrives. Some then cross the
sparsification threshold in either direction, so an incremental update silently
rewrites the base graph's topology.

Freezing the transform is therefore not a refinement of incremental ingestion,
it is what makes it correct. A generation records the statistics it actually
used; a delta built against that generation reuses them rather than deriving new
ones.

Recording `edge_sparsify_threshold` and the posting caps is not the same thing.
Those are configuration — inputs to the build. What matters is the min and max
observed, which are outputs of it.

OUT OF RANGE. A delta value above the frozen maximum is clipped to 1.0 rather
than widening the range, because widening is precisely the rescale being
avoided. Excursions are counted, so drift is visible and can justify a
deliberate full rebuild instead of happening silently.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

#: Bumped when the normalisation formula changes, so a generation built under an
#: older rule is not reused under a newer one.
TRANSFORM_VERSION = 1


@dataclass
class SignalTransform:
    """The min-max transform one signal was built with."""
    lo: float
    hi: float
    version: int = TRANSFORM_VERSION
    #: What `_minmax_normalize` does when every raw score is equal: it returns 1.0
    #: for all edges rather than dividing by a zero span. Recorded so a delta
    #: reproduces it instead of rediscovering it.
    constant_range_value: float = 1.0
    clipped_low: int = 0
    clipped_high: int = 0

    @property
    def is_constant(self) -> bool:
        return self.hi <= self.lo

    def apply(self, score: float) -> float:
        """Normalise one raw score under this frozen transform."""
        if self.is_constant:
            return self.constant_range_value
        if score <= self.lo:
            if score < self.lo:
                self.clipped_low += 1
            return 0.0
        if score >= self.hi:
            if score > self.hi:
                self.clipped_high += 1
            return 1.0
        return (score - self.lo) / (self.hi - self.lo)

    @classmethod
    def fit(cls, scores) -> "SignalTransform":
        vals = list(scores)
        if not vals:
            return cls(lo=0.0, hi=0.0)
        return cls(lo=float(min(vals)), hi=float(max(vals)))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SignalTransform":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def fresh(self) -> "SignalTransform":
        """A copy with the excursion counters reset.

        `apply` mutates them, so reusing one frozen set across two builds would
        accumulate counts and misreport how far a single delta drifted.
        """
        return SignalTransform(lo=self.lo, hi=self.hi, version=self.version,
                               constant_range_value=self.constant_range_value)


@dataclass
class TransformSet:
    """Every signal's frozen transform, as carried in a generation manifest."""
    signals: dict[str, SignalTransform] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"version": TRANSFORM_VERSION,
                "signals": {k: v.to_dict() for k, v in self.signals.items()}}

    @classmethod
    def from_dict(cls, d: dict | None) -> "TransformSet | None":
        if not d or "signals" not in d:
            return None
        if d.get("version") != TRANSFORM_VERSION:
            logger.warning(
                "Frozen transforms are version %s but this build uses %s — "
                "ignoring them and refitting. A delta against this generation "
                "would not be on the same scale.", d.get("version"), TRANSFORM_VERSION)
            return None
        return cls({k: SignalTransform.from_dict(v) for k, v in d["signals"].items()})

    def fresh(self) -> "TransformSet":
        """A copy whose per-signal excursion counters start at zero."""
        return TransformSet({k: v.fresh() for k, v in self.signals.items()})

    def excursions(self) -> dict[str, tuple[int, int]]:
        """Per signal, how many delta scores fell outside the frozen range.

        Non-zero means the corpus has drifted past what the base generation saw.
        Those edges are clipped rather than rescaling the base, so this is the
        signal that a deliberate full rebuild is due.
        """
        return {k: (t.clipped_low, t.clipped_high) for k, t in self.signals.items()
                if t.clipped_low or t.clipped_high}
