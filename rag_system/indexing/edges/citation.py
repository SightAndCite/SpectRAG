from __future__ import annotations
import re
from collections import defaultdict
from rag_system.models import Chunk
from rag_system.indexing.edges.base import EdgeExtractor, RawEdge
from rag_system.indexing.edges.postings import edges_from_postings
from config import IndexingConfig

_CITE_BRACKET_RE = re.compile(r"\[(\d+(?:[,;]\s*\d+)*)\]")
_DOI_RE = re.compile(r"doi:\s*(10\.\d{4,}/\S+)", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s>\"']+")


class CitationEdgeExtractor(EdgeExtractor):
    """
    Edges between chunks that share citation keys (bracket refs, DOIs, URLs).
    Directional signal (one cites the other) is symmetrized in the graph builder.

    Fan-out is bounded on both sides: a key present in too large a fraction of
    the corpus is dropped, and a chunk contributes only its rarest keys. A
    reference-list page emits every bracket ref, DOI and URL it contains, so
    bibliographies would otherwise form one dense mutually-linked block.
    """

    def __init__(self, cfg: IndexingConfig | None = None) -> None:
        cfg = cfg or IndexingConfig()
        self._max_df_ratio  = cfg.shared_key_max_df_ratio
        self._max_df_abs    = cfg.shared_key_max_df_abs
        self._df_floor      = cfg.shared_key_df_floor
        self._max_per_chunk = cfg.shared_key_max_per_chunk
        self._max_neighbors = cfg.shared_key_max_neighbors
        self._idf           = cfg.shared_key_idf_weighting

    def extract(self, chunks: list[Chunk]) -> list[RawEdge]:
        key_to_chunks: dict[str, list[int]] = defaultdict(list)
        for i, chunk in enumerate(chunks):
            for key in self._citation_keys(chunk.text):
                key_to_chunks[key].append(i)

        return edges_from_postings(
            key_to_chunks, len(chunks),
            min_df=2,
            max_df_ratio=self._max_df_ratio,
            max_df_abs=self._max_df_abs,
            df_floor=self._df_floor,
            max_per_chunk=self._max_per_chunk,
            max_neighbors=self._max_neighbors,
            idf_weighting=self._idf,
            label="citation",
        )

    @staticmethod
    def _citation_keys(text: str) -> set[str]:
        keys: set[str] = set()
        for m in _CITE_BRACKET_RE.finditer(text):
            for num in re.split(r"[,;]\s*", m.group(1)):
                keys.add(f"ref:{num.strip()}")
        for m in _DOI_RE.finditer(text):
            keys.add(f"doi:{m.group(1).lower()}")
        for url in _URL_RE.findall(text):
            keys.add(f"url:{url.rstrip('.,;)')}")
        return keys
