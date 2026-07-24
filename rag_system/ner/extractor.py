from __future__ import annotations
import logging
from typing import NamedTuple

import ollama
import spacy

from config import NERConfig, LanguageConfig, OllamaConfig
from prompts import LLM_NER_PROMPT
from rag_system.language.detector import spacy_model_for, language_name

logger = logging.getLogger(__name__)


class Entity(NamedTuple):
    text: str
    label: str
    normalized: str     # lowercased + stripped, used as graph edge key


class HybridNERExtractor:
    """
    Per-language NER with three tiers:
      1. Language-specific spaCy model (e.g. fr_core_news_lg for French)
      2. Multilingual spaCy fallback (xx_core_web_sm) if no dedicated model installed
      3. LLM-based NER (Ollama) when spaCy is unavailable for the language and
         cfg_lang.use_llm_ner_fallback is True

    Models are loaded lazily and cached for reuse across chunks.
    GLiNER is opt-in via cfg_ner.use_gliner (English / multilingual).
    """

    def __init__(
        self,
        cfg_ner: NERConfig,
        cfg_lang: LanguageConfig | None = None,
        cfg_ollama: OllamaConfig | None = None,
    ) -> None:
        self._cfg_ner    = cfg_ner
        self._cfg_lang   = cfg_lang or LanguageConfig()
        self._cfg_ollama = cfg_ollama
        self._spacy_cache: dict[str, spacy.language.Language | None] = {}
        self._gliner = None

        if cfg_ner.use_gliner:
            self._load_gliner()

        if cfg_ollama is not None:
            self._ollama_client = ollama.Client(host=cfg_ollama.base_url)
        else:
            self._ollama_client = None

    # spaCy helpers

    def _load_spacy(self, model_name: str) -> spacy.language.Language | None:
        try:
            return spacy.load(model_name, disable=["parser", "lemmatizer"])
        except OSError:
            return None

    def _get_spacy(self, lang_code: str) -> spacy.language.Language | None:
        if lang_code in self._spacy_cache:
            return self._spacy_cache[lang_code]

        preferred = spacy_model_for(lang_code)
        nlp = self._load_spacy(preferred)
        if nlp is None and preferred != "xx_core_web_sm":
            logger.warning(
                "spaCy model '%s' not installed for lang '%s', trying xx_core_web_sm",
                preferred, lang_code,
            )
            nlp = self._load_spacy("xx_core_web_sm")
        if nlp is None:
            logger.warning("No spaCy model available for lang '%s'", lang_code)

        self._spacy_cache[lang_code] = nlp
        return nlp

    # GLiNER helper

    def _load_gliner(self) -> None:
        try:
            from gliner import GLiNER  # noqa: PLC0415
            logger.info("Loading GLiNER model: %s", self._cfg_ner.gliner_model)
            self._gliner = GLiNER.from_pretrained(self._cfg_ner.gliner_model)
            logger.info("GLiNER ready.")
        except Exception as exc:
            logger.warning("GLiNER failed to load, falling back to spaCy-only: %s", exc)
            self._gliner = None

    # LLM NER fallback

    def _llm_extract(self, text: str, lang_code: str) -> list[Entity]:
        if self._ollama_client is None or self._cfg_ollama is None:
            return []
        lang = language_name(lang_code)
        prompt = LLM_NER_PROMPT.format(
            language=lang,
            text=text[: self._cfg_lang.llm_ner_max_text_chars],
        )
        try:
            resp = self._ollama_client.generate(
                model=self._cfg_ollama.llm_model,
                prompt=prompt,
                options={
                    "temperature": self._cfg_lang.llm_ner_temperature,
                    "num_predict": self._cfg_lang.llm_ner_max_tokens,
                },
            )
            entities: list[Entity] = []
            for line in resp.response.strip().splitlines():
                line = line.strip()
                if "|" not in line:
                    continue
                parts = line.split("|", 1)
                ent_text  = parts[0].strip()
                ent_label = parts[1].strip().upper()
                if ent_text:
                    entities.append(Entity(ent_text, ent_label, ent_text.lower().strip()))
            return entities
        except Exception as exc:
            logger.warning("LLM NER failed for lang '%s': %s", lang_code, exc)
            return []

    # Public API

    def extract(self, text: str, language: str = "en") -> list[Entity]:
        seen: dict[tuple[str, str], Entity] = {}

        nlp = self._get_spacy(language)
        if nlp is not None:
            for ent in nlp(text).ents:
                key = (ent.text.lower().strip(), ent.label_)
                seen[key] = Entity(ent.text, ent.label_, ent.text.lower().strip())
        elif self._cfg_lang.use_llm_ner_fallback:
            for ent in self._llm_extract(text, language):
                key = (ent.normalized, ent.label)
                seen[key] = ent

        # GLiNER — domain-specific extras (opt-in, language-agnostic model)
        if self._gliner is not None:
            for ent in self._gliner.predict_entities(
                text,
                self._cfg_ner.gliner_entity_types,
                threshold=self._cfg_ner.gliner_threshold,
                flat_ner=self._cfg_ner.gliner_flat_ner,
            ):
                key = (ent["text"].lower().strip(), ent["label"].upper())
                if key not in seen:
                    seen[key] = Entity(
                        ent["text"],
                        ent["label"].upper(),
                        ent["text"].lower().strip(),
                    )

        return list(seen.values())
