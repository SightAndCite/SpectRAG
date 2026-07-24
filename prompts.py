"""
All prompt templates for the Graph-Spectral RAG system.

Import from this module only — never embed prompt strings in logic files.
Each template uses str.format() keyword substitution.
"""
from __future__ import annotations


# Stage 5: answer generation

# Placeholders: {language}
GENERATION_SYSTEM = (
    "You are a knowledgeable, grounded analyst.\n"
    "Answer the user's question in {language} by synthesizing and connecting "
    "information across the provided context passages. Draw out themes, "
    "relationships, and implications that the passages support, even when no "
    "single passage states the answer verbatim — combine partial evidence into "
    "a coherent, multi-perspective answer.\n"
    "Ground every claim in the context: do not introduce specific facts, names, "
    "or figures that are not supported by the passages, and do not fabricate. "
    "Only if the passages are truly unrelated to the question should you say the "
    "context is insufficient; otherwise give the most complete answer the "
    "context supports."
)

# Placeholders: {i}, {source}, {pos}, {text}
GENERATION_CONTEXT_BLOCK = (
    "--- Context {i} (source: {source}, pos: {pos}) ---\n{text}\n"
)

# Placeholder: {text}. Extracts cross-document concept keys for the entity/concept
# edge signal — the point is CANONICAL phrases two passages about the same idea
# both produce, so shared concepts link related chunks even when wording differs.
CONCEPT_EXTRACTION_PROMPT = (
    "Extract the 5-10 most important domain concepts, topics, methods, or named "
    "entities from the passage below. Return canonical, lowercase short noun "
    "phrases (2-4 words each) that another passage discussing the same idea would "
    "also produce — the goal is that shared concepts link related passages. "
    "Prefer specific domain terms over generic filler.\n\n"
    "Passage:\n{text}\n\n"
    'Respond as JSON: {{"concepts": ["concept one", "concept two", ...]}}'
)

# Placeholders: {system}, {context}, {question}
GENERATION_PROMPT = "{system}\n\n{context}\nQuestion: {question}\n\nAnswer:"


# Edge signal: utility-question generation

# Placeholders: {n}, {language}, {text}
UTILITY_QUESTION_PROMPT = (
    "Generate {n} concise questions in {language} that this passage directly answers. "
    "Output only the questions, one per line, no numbering or preamble.\n\n"
    "Passage:\n{text}"
)


# NER fallback: LLM-based entity extraction

# Placeholders: {language}, {text}
LLM_NER_PROMPT = (
    "Extract named entities from the text below written in {language}.\n"
    "Return each entity on its own line in the format: TEXT|TYPE\n"
    "Types: PERSON, ORG, LOCATION, PRODUCT, DATE, EVENT, TECHNOLOGY, CONCEPT\n"
    "Output only the entity lines, no preamble or explanation.\n\n"
    "Text:\n{text}"
)
