from __future__ import annotations
import logging
from collections import defaultdict
from rag_system.models import Chunk
from rag_system.indexing.edges.base import EdgeExtractor, RawEdge
from rag_system.indexing.edges.postings import edges_from_postings
from config import IndexingConfig, LanguageConfig, NERConfig, OllamaConfig, OpenAIConfig

logger = logging.getLogger(__name__)


class EntityEdgeExtractor(EdgeExtractor):
    """Cross-chunk edges via shared entities/concepts (the primary multi-hop link).

    Terms come from either spaCy/GLiNER NER ("ner") or LLM concept extraction
    ("llm_concepts", see llm_concept.py) per `entity_extraction_mode`. The edge
    logic is identical — chunks sharing a term get linked — only the term source
    changes."""

    def __init__(
        self,
        cfg: IndexingConfig,
        ner_cfg: NERConfig,
        lang_cfg: LanguageConfig | None = None,
        ollama_cfg: OllamaConfig | None = None,
        openai_cfg: OpenAIConfig | None = None,
    ) -> None:
        self._min_freq     = cfg.entity_min_freq
        self._max_df_ratio = cfg.shared_key_max_df_ratio
        self._max_df_abs   = cfg.shared_key_max_df_abs
        self._df_floor     = cfg.shared_key_df_floor
        self._max_per_chunk = cfg.shared_key_max_per_chunk
        self._max_neighbors = cfg.shared_key_max_neighbors
        self._idf          = cfg.shared_key_idf_weighting
        self._log_interval = cfg.entity_log_interval
        self._mode         = cfg.entity_extraction_mode
        if self._mode == "llm_concepts":
            from rag_system.indexing.edges.llm_concept import LLMConceptExtractor
            self._concept = LLMConceptExtractor(openai_cfg, cfg)
            self._ner = None
        else:
            # Imported here so llm_concepts mode does not require spaCy.
            from rag_system.ner.extractor import HybridNERExtractor
            self._ner = HybridNERExtractor(ner_cfg, lang_cfg, ollama_cfg)
            self._concept = None

    def _terms(self, chunk: Chunk) -> list[str]:
        if self._concept is not None:
            return self._concept.extract(chunk.text)
        return [ent.normalized for ent in self._ner.extract(chunk.text, language=chunk.language)]

    def extract(self, chunks: list[Chunk]) -> list[RawEdge]:
        entity_to_chunks: dict[str, list[int]] = defaultdict(list)
        if self._concept is not None:
            # Parallel LLM concept extraction (llm_parallel_workers threads);
            # cache hits return instantly, misses retry transient API errors.
            per_chunk = self._concept.extract_batch(
                [c.text for c in chunks],
                progress_cb=lambda done, total: logger.info(
                    "Concept/entity extraction: %d / %d chunks", done, total),
                progress_every=self._log_interval,
            )
            for i, terms in enumerate(per_chunk):
                for term in terms:
                    entity_to_chunks[term].append(i)
            self._concept.close()
        else:
            for i, chunk in enumerate(chunks):
                if (i + 1) % self._log_interval == 0:
                    logger.info("Concept/entity extraction: %d / %d chunks", i + 1, len(chunks))
                for term in self._terms(chunk):
                    entity_to_chunks[term].append(i)

        return edges_from_postings(
            entity_to_chunks, len(chunks),
            min_df=self._min_freq,
            max_df_ratio=self._max_df_ratio,
            max_df_abs=self._max_df_abs,
            df_floor=self._df_floor,
            max_per_chunk=self._max_per_chunk,
            max_neighbors=self._max_neighbors,
            idf_weighting=self._idf,
            label="entity",
        )
