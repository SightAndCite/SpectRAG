"""Immutable index generations with atomic publication.

Artifacts used to be written independently into one directory — chunks, vectors,
the FAISS index, the lexical index, the question index, the graph — and readiness
was decided by whether `chunks.pkl` existed. A build interrupted between writes
therefore left a half-published index that read as ready, and a rebuild wrote
over the artifacts a live reader was using. `vectors.npy` from one build beside
`lexical/` from another does not fail; it silently returns wrong results.

Each build now writes into `generations/<id>/` and is published by atomically
replacing a manifest pointer. Publication is the single `os.replace` of that
manifest, so a reader sees either the old generation or the new one, never a
mixture. Validation runs before the swap: every declared artifact must exist and
the per-artifact counts must agree, so an incomplete build cannot become active.

Readers resolve the manifest once and hold the resulting path for the whole
request. A publication mid-request cannot move the files underneath them, and
the previous generation is retained until it is pruned — which also means the
current generation keeps serving while the next one builds.

Frozen transforms live in the manifest too. Corpus-global min/max edge
normalisation means a new extreme rescales every stored edge, so incremental
updates are impossible while those statistics float. Recording them per
generation is what later lets a delta be scored on the same scale as the base.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST = "manifest.json"
GENERATIONS = "generations"
_STAGING = ".staging"
#: Key marker for an artifact whose row count must equal chunk_count.
_ROW_COUNTED = "chunk:"


@dataclass
class Manifest:
    """What one generation contains and what produced it."""
    generation_id: str
    created_at: float
    chunk_count: int = 0
    artifacts: dict[str, int] = field(default_factory=dict)   # relative path -> count
    embedding_model: str = ""
    vector_index_type: str = "flat"
    uq_prefix_role: str = ""
    # Score transforms frozen at build time. Incremental updates cannot exist
    # while corpus-global extrema float, because a new extreme silently rescales
    # every edge already published.
    score_transforms: dict = field(default_factory=dict)
    schema_version: int = 1

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, blob: str) -> "Manifest":
        data = json.loads(blob)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


class GenerationStore:
    """Resolves, stages and publishes index generations under one store root."""

    def __init__(self, root: Path | str, keep: int = 2) -> None:
        self.root = Path(root)
        # Two on disk: the active generation and its predecessor, for rollback
        # and for readers still draining. That is the minimum an atomic swap
        # requires — deleting the old one at publish would pull files out from
        # under a pinned reader.
        self.keep = max(1, keep)

    # Reading

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST

    def active(self) -> tuple[Path, Manifest] | None:
        """Resolve the published generation. Returns None if nothing is published.

        Callers should resolve once per request and reuse the path: that is what
        makes a concurrent publication invisible to an in-flight read.
        """
        mp = self.manifest_path
        if not mp.exists():
            return None
        try:
            manifest = Manifest.from_json(mp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, TypeError, OSError) as exc:
            logger.error("Unreadable manifest at %s (%s)", mp, exc)
            return None
        path = self.root / GENERATIONS / manifest.generation_id
        if not path.is_dir():
            logger.error("Manifest points at missing generation %s",
                         manifest.generation_id)
            return None
        return path, manifest

    def has_published(self) -> bool:
        return self.active() is not None

    # Writing

    def stage(self) -> tuple[Path, str]:
        """A fresh directory to build into. Not visible to readers until published."""
        # Microseconds in the id so lexical order matches creation order. Two
        # builds within the same second would otherwise sort by their random
        # suffix, which made "newest first" wrong.
        now = time.time()
        stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime(now))
        gen_id = f"{stamp}.{int(now % 1 * 1e6):06d}-{uuid.uuid4().hex[:8]}"
        path = self.root / GENERATIONS / _STAGING / gen_id
        path.mkdir(parents=True, exist_ok=True)
        return path, gen_id

    def publish(self, staged: Path, manifest: Manifest) -> Path:
        """Validate, move into place, then swap the manifest atomically."""
        self._validate(staged, manifest)

        final = self.root / GENERATIONS / manifest.generation_id
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            shutil.rmtree(final, ignore_errors=True)
        os.replace(staged, final)

        # The one operation that makes the new generation visible. Everything
        # before it is invisible to readers; everything after is complete.
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(manifest.to_json(), encoding="utf-8")
        os.replace(tmp, self.manifest_path)

        logger.info("Published generation %s (%d chunks, %d artifacts)",
                    manifest.generation_id, manifest.chunk_count, len(manifest.artifacts))
        self.prune()
        return final

    def _validate(self, staged: Path, manifest: Manifest) -> None:
        """Every declared artifact must exist, and counts must agree.

        This is what stops a build that died between writes from being published:
        the check runs against the staging directory, before anything readers can
        see is touched.
        """
        if not manifest.artifacts:
            # An empty declaration would make every check vacuous. The manifest
            # must state what the build produced, captured when save() completed,
            # so that a file going missing afterwards is detectable.
            raise ValueError(
                f"Generation {manifest.generation_id} declares no artifacts — "
                f"nothing to validate. Not publishing."
            )

        # A key may carry a "chunk:" marker meaning "this artifact must also hold
        # exactly chunk_count rows"; the marker is not part of the path.
        def rel_path(key: str) -> str:
            return key[len(_ROW_COUNTED):] if key.startswith(_ROW_COUNTED) else key

        missing = [rel_path(k) for k in manifest.artifacts
                   if not (staged / rel_path(k)).exists()]
        if missing:
            raise ValueError(
                f"Generation {manifest.generation_id} is incomplete — missing "
                f"{missing}. Not publishing."
            )
        bad = {rel_path(k): n for k, n in manifest.artifacts.items()
               if k.startswith(_ROW_COUNTED) and n >= 0 and n != manifest.chunk_count}
        if bad:
            raise ValueError(
                f"Generation {manifest.generation_id} is inconsistent: {bad} rows "
                f"against {manifest.chunk_count} chunks. Not publishing."
            )

    def prune(self) -> list[str]:
        """Remove generations beyond the retention count, newest first."""
        gens_dir = self.root / GENERATIONS
        if not gens_dir.is_dir():
            return []
        act = self.active()
        keep_id = act[1].generation_id if act else None
        others = sorted((d for d in gens_dir.iterdir()
                         if d.is_dir() and d.name != _STAGING and d.name != keep_id),
                        key=lambda d: d.name, reverse=True)
        # The active generation is always first and COUNTS toward the retention
        # budget. Exempting it instead let keep=2 leave three on disk.
        ordered = ([gens_dir / keep_id] if keep_id else []) + others
        removed = []
        for d in ordered[self.keep:]:
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d.name)
        if removed:
            logger.info("Pruned generations: %s", ", ".join(removed))
        return removed

    def discard_staging(self) -> None:
        """Drop staged builds that never published (a crashed or failed run)."""
        staging = self.root / GENERATIONS / _STAGING
        if staging.is_dir():
            shutil.rmtree(staging, ignore_errors=True)
