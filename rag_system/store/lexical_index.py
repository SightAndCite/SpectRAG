"""Persistent inverted index with exact BM25Okapi scoring.

`rank-bm25` keeps the whole tokenised corpus alive as Python lists plus one
term-frequency dict per document, and scores every document on every query.
Measured on the harness that is ~15 kB per chunk — about 15 GB at 1M chunks,
built inside the first query after a restart.

BM25Okapi's score for a document is a sum over query terms, and a term the
document does not contain contributes exactly zero (its term frequency is 0, so
the numerator is 0). Visiting only the postings of the query's terms is
therefore *arithmetically identical* to scoring the full corpus, not an
approximation — which is what lets this replace rank-bm25 with no retrieval
change at all.

Layout on disk (all memory-mappable):

    lexical/ids.npy      int32   concatenated chunk ids, one block per term
    lexical/tfs.npy      int32   term frequencies, parallel to ids
    lexical/offsets.npy  int64   block boundaries, len(vocab) + 1
    lexical/idf.npy      float64 per-term IDF, already floored
    lexical/doc_len.npy  int32   token count per chunk
    lexical/vocab.json   term -> row
    lexical/meta.json    corpus_size, avgdl, k1, b, epsilon
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_IDS, _TFS, _OFFSETS = "ids.npy", "tfs.npy", "offsets.npy"
_IDF, _DOC_LEN = "idf.npy", "doc_len.npy"
_VOCAB, _META = "vocab.json", "meta.json"


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens for BM25 (lexical, no stemming)."""
    return _TOKEN_RE.findall(text.lower())


class LexicalIndex:
    """Inverted index reproducing `rank_bm25.BM25Okapi` scores exactly."""

    def __init__(
        self,
        vocab: dict[str, int],
        offsets: np.ndarray,
        ids: np.ndarray,
        tfs: np.ndarray,
        idf: np.ndarray,
        doc_len: np.ndarray,
        corpus_size: int,
        avgdl: float,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        self.vocab, self.offsets, self.ids, self.tfs = vocab, offsets, ids, tfs
        self.idf, self.doc_len = idf, doc_len
        self.corpus_size, self.avgdl = corpus_size, avgdl
        self.k1, self.b, self.epsilon = k1, b, epsilon
        # Per-document denominator term, k1 * (1 - b + b*len/avgdl). The operator
        # association mirrors BM25Okapi.get_scores exactly -- (b*len)/avgdl, not
        # b*(len/avgdl) -- because float rounding differs between them and the
        # point of this class is to be bit-identical, not merely close.
        dl = np.asarray(doc_len, dtype=np.float64)
        denom = (1.0 - b + b * dl / avgdl) if avgdl > 0 else \
            np.full(corpus_size, 1.0 - b, dtype=np.float64)
        self._k1_norm = k1 * denom

    # Build

    @classmethod
    def build(cls, texts: list[str], k1: float = 1.5, b: float = 0.75,
              epsilon: float = 0.25) -> "LexicalIndex":
        n = len(texts)
        # Plain dict, not defaultdict: insertion order is first-appearance order,
        # which is the order BM25Okapi's `nd` uses when it sums IDF values. Float
        # addition is not associative, so summing in a different order shifts
        # average_idf, and with it every floored (negative-IDF) term.
        postings: dict[str, dict[int, int]] = {}
        doc_len = np.zeros(n, dtype=np.int32)
        for i, text in enumerate(texts):
            toks = tokenize(text)
            doc_len[i] = len(toks)
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            for t, c in counts.items():
                postings.setdefault(t, {})[i] = c

        terms = list(postings)          # first-appearance order, as above
        vocab = {t: r for r, t in enumerate(terms)}
        total = sum(len(postings[t]) for t in terms)
        offsets = np.zeros(len(terms) + 1, dtype=np.int64)
        ids = np.empty(total, dtype=np.int32)
        tfs = np.empty(total, dtype=np.int32)
        pos = 0
        for r, t in enumerate(terms):
            block = sorted(postings[t].items())
            offsets[r] = pos
            for cid, tf in block:
                ids[pos], tfs[pos] = cid, tf
                pos += 1
        offsets[len(terms)] = pos

        # BM25Okapi's IDF, including its floor: terms in more than half the corpus
        # get a negative raw IDF and are replaced by epsilon * average_idf.
        raw = np.array(
            [math.log(n - len(postings[t]) + 0.5) - math.log(len(postings[t]) + 0.5)
             for t in terms],
            dtype=np.float64,
        )
        idf = raw.copy()
        if len(terms):
            # Sequential accumulation in Python, matching BM25Okapi. np.mean uses
            # pairwise summation and lands a few ulps away, which then propagates
            # into every floored term.
            idf_sum = 0.0
            for v in raw.tolist():
                idf_sum += v
            eps = epsilon * (idf_sum / len(raw))
            idf[raw < 0] = eps

        avgdl = float(doc_len.sum()) / n if n else 0.0
        logger.info("LexicalIndex: %d chunks, %d terms, %d postings", n, len(terms), total)
        return cls(vocab, offsets, ids, tfs, idf, doc_len, n, avgdl, k1, b, epsilon)

    # Scoring

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        """Dense (N,) BM25 scores, identical to BM25Okapi.get_scores.

        Only the postings of terms the query actually uses are read. A repeated
        query term is applied twice, matching BM25Okapi, which iterates the query
        list rather than a set.
        """
        out = np.zeros(self.corpus_size, dtype=np.float64)
        for term in query_tokens:
            row = self.vocab.get(term)
            if row is None:
                continue                      # BM25Okapi: idf.get(q) or 0 -> no effect
            lo, hi = int(self.offsets[row]), int(self.offsets[row + 1])
            if lo == hi:
                continue
            ids = np.asarray(self.ids[lo:hi], dtype=np.int64)
            tf = np.asarray(self.tfs[lo:hi], dtype=np.float64)
            idf = float(self.idf[row])
            # idf * (num / den), not (idf * num) / den -- same reason as above.
            out[ids] += idf * (tf * (self.k1 + 1) / (tf + self._k1_norm[ids]))
        return out

    # Persistence

    def save(self, directory: Path | str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / _IDS, self.ids)
        np.save(d / _TFS, self.tfs)
        np.save(d / _OFFSETS, self.offsets)
        np.save(d / _IDF, self.idf)
        np.save(d / _DOC_LEN, self.doc_len)
        (d / _VOCAB).write_text(json.dumps(self.vocab), encoding="utf-8")
        (d / _META).write_text(json.dumps({
            "corpus_size": self.corpus_size, "avgdl": self.avgdl,
            "k1": self.k1, "b": self.b, "epsilon": self.epsilon,
        }), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path | str) -> "LexicalIndex | None":
        d = Path(directory)
        if not (d / _META).exists():
            return None
        meta = json.loads((d / _META).read_text(encoding="utf-8"))
        vocab = json.loads((d / _VOCAB).read_text(encoding="utf-8"))
        mm = lambda name: np.load(d / name, mmap_mode="r")   # noqa: E731
        return cls(
            vocab, mm(_OFFSETS), mm(_IDS), mm(_TFS),
            np.asarray(np.load(d / _IDF)), np.asarray(np.load(d / _DOC_LEN)),
            meta["corpus_size"], meta["avgdl"],
            meta["k1"], meta["b"], meta["epsilon"],
        )

    @staticmethod
    def directory_name() -> str:
        return "lexical"
