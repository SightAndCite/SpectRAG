from __future__ import annotations
import logging
from openai import OpenAI
from rag_system.models import Chunk
from rag_system.language.detector import detect_language, language_name
from config import Config
from prompts import GENERATION_SYSTEM, GENERATION_CONTEXT_BLOCK

logger = logging.getLogger(__name__)


class Generator:
    """Stage 5 — formats retrieved context and calls the LLM (OpenAI gpt-4o-mini)."""

    def __init__(self, cfg: Config) -> None:
        self._openai   = cfg.openai
        self._lang_cfg = cfg.language
        if not self._openai.api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set — the final-answer generator requires it. "
                "Export it before starting the server/CLI."
            )
        self._client = OpenAI(
            api_key=self._openai.api_key,
            base_url=self._openai.base_url,
            timeout=self._openai.request_timeout,
        )

    def generate(self, question: str, context_chunks: list[Chunk]) -> str:
        lang_code = detect_language(question, fallback=self._lang_cfg.fallback_language)
        lang      = language_name(lang_code)
        logger.debug("Query language detected: %s (%s)", lang, lang_code)

        # Group context by source document, with a header per source, so the
        # generator can see — and synthesize across — multiple sources. Applied
        # identically to SpectRAG and the NaiveRAG baseline (both share this
        # generator), so it stays a fair formatting change: it simply reflects
        # each system's real source spread (SpectRAG's graph expansion typically
        # spans more documents than naive top-k).
        groups: dict[str, list[Chunk]] = {}
        for chunk in context_chunks:
            src = chunk.metadata.get("source", chunk.doc_id)
            groups.setdefault(src, []).append(chunk)

        blocks: list[str] = []
        n = 0
        for src, group_chunks in groups.items():
            blocks.append(f"=== Source: {src} ===")
            for chunk in group_chunks:
                n += 1
                blocks.append(GENERATION_CONTEXT_BLOCK.format(
                    i=n, source=src, pos=chunk.position, text=chunk.text,
                ))
        context_text = "\n".join(blocks)

        system = GENERATION_SYSTEM.format(language=lang)
        user   = f"{context_text}\nQuestion: {question}\n\nAnswer:"

        resp = self._client.chat.completions.create(
            model=self._openai.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=self._openai.temperature,
            max_tokens=self._openai.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    def close(self) -> None:
        pass  # OpenAI client manages its own HTTP pool
