"""Persisted search index over chunks' utility questions.

Each chunk carries LLM-generated "what question does this answer?" strings, built
at index time for the utility-question edges. At query time they are matched
against the user's question — query-shaped surrogates of the chunk, which bridges
the abstractive-query/passage gap that plain dense retrieval struggles with.

That index used to be constructed inside the *first query* after a restart or
corpus switch: every stored question embedded and a fresh flat index built, while
a user waited. Two questions per chunk at 1M is 2M embeddings and 6.14 GB — not a
slow first query so much as one that never returns. It also published its cache
key before the artifact, so a concurrent request during the build silently
answered without utility-question seeds.

PREFIX ROLE. nomic-embed-text is trained with task prefixes, and the two
consumers of these questions historically disagreed: edge construction embeds
them as *documents*, while the query-side seed index embeds them as *queries*.
Both are internally consistent, because each only ever compares within its own
space — so it was never a live defect, but persisting a single artifact forces
the choice into the open. This index stores the QUERY role, which is what the
serving path already used, so persisting it changes no result. The role is
recorded in the manifest and validated on load, so an artifact built under one
policy can never be served under another.

Unifying the two roles would halve build-time embedding cost and remove an
artifact, and is the nomic-intended asymmetry (query vs document). It is a
measurable retrieval change and belongs with F12, not here.

Layout:

    questions/index.faiss    flat inner-product index over question vectors
    questions/to_chunk.npy   int32, question row -> chunk index
    questions/meta.json      prefix role, model, dim, counts
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)

_INDEX, _MAPPING, _META = "index.faiss", "to_chunk.npy", "meta.json"


class QuestionIndex:
    """Flat inner-product index over utility questions, with its chunk mapping."""

    def __init__(self, index: faiss.Index, to_chunk: np.ndarray,
                 prefix_role: str, model: str) -> None:
        self.index = index
        self.to_chunk = to_chunk
        self.prefix_role = prefix_role
        self.model = model

    def __len__(self) -> int:
        return int(self.to_chunk.shape[0])

    @classmethod
    def build(cls, chunks, embedder, prefix_role: str = "query",
              model: str = "") -> "QuestionIndex | None":
        """Embed every chunk's utility questions once, at index time.

        Returns None when no chunk carries questions, which is the normal case
        when utility-question generation is disabled.
        """
        questions: list[str] = []
        mapping: list[int] = []
        for ci, chunk in enumerate(chunks):
            for q in chunk.utility_questions:
                questions.append(q)
                mapping.append(ci)
        if not questions:
            return None

        embs = embedder.embed(questions, kind=prefix_role).astype(np.float32)
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        logger.info("QuestionIndex: %d questions over %d chunks (%s role)",
                    len(questions), len(chunks), prefix_role)
        return cls(index, np.asarray(mapping, dtype=np.int32), prefix_role, model)

    def search(self, q_emb: np.ndarray, k: int):
        k = min(k, len(self))
        if k <= 0:
            return np.empty((1, 0), dtype=np.int64)
        _, idx = self.index.search(q_emb, k)
        return idx

    def save(self, directory: Path | str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / _INDEX))
        np.save(d / _MAPPING, self.to_chunk)
        (d / _META).write_text(json.dumps({
            "prefix_role": self.prefix_role, "model": self.model,
            "dim": self.index.d, "questions": len(self),
        }), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path | str, expect_role: str = "query",
             expect_model: str = "") -> "QuestionIndex | None":
        d = Path(directory)
        if not (d / _META).exists():
            return None
        meta = json.loads((d / _META).read_text(encoding="utf-8"))
        # An artifact built under a different prefix role or embedder is not
        # interchangeable: the vectors live in a different space, and reusing them
        # would silently degrade seed matching rather than fail.
        if meta.get("prefix_role") != expect_role:
            logger.warning(
                "Question index was built with prefix role %r but %r is expected — "
                "ignoring it and rebuilding. Re-index to persist the current role.",
                meta.get("prefix_role"), expect_role)
            return None
        if expect_model and meta.get("model") and meta["model"] != expect_model:
            logger.warning(
                "Question index was built with model %r but %r is configured — "
                "ignoring it. Re-index to persist vectors for the current model.",
                meta["model"], expect_model)
            return None
        index = faiss.read_index(str(d / _INDEX))
        to_chunk = np.load(d / _MAPPING)
        if index.ntotal != len(to_chunk):
            raise ValueError(
                f"Question index holds {index.ntotal} vectors but the mapping has "
                f"{len(to_chunk)} entries — the artifact is inconsistent, rebuild it."
            )
        return cls(index, to_chunk, meta["prefix_role"], meta.get("model", ""))

    @staticmethod
    def directory_name() -> str:
        return "questions"
