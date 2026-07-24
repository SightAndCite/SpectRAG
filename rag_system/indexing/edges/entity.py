from __future__ import annotations
import logging
from collections import defaultdict
from rag_system.models import Chunk
from rag_system.indexing.edges.base import EdgeExtractor, RawEdge
from rag_system.ner.extractor import HybridNERExtractor
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
        self._log_interval = cfg.entity_log_interval
        self._mode         = cfg.entity_extraction_mode
        if self._mode == "llm_concepts":
            from rag_system.indexing.edges.llm_concept import LLMConceptExtractor
            self._concept = LLMConceptExtractor(openai_cfg, cfg)
            self._ner = None
        else:
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

        edge_counts: dict[tuple[int, int], float] = {}
        for chunk_idxs in entity_to_chunks.values():
            if len(chunk_idxs) < self._min_freq:
                continue
            for a in range(len(chunk_idxs)):
                for b in range(a + 1, len(chunk_idxs)):
                    key = (chunk_idxs[a], chunk_idxs[b])
                    edge_counts[key] = edge_counts.get(key, 0.0) + 1.0

        if not edge_counts:
            return []

        max_count = max(edge_counts.values())
        return [RawEdge(i, j, count / max_count) for (i, j), count in edge_counts.items()]
