from __future__ import annotations

try:
    from langdetect import detect, LangDetectException
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

NLTK_LANGUAGE_MAP: dict[str, str] = {
    "da": "danish",
    "nl": "dutch",
    "en": "english",
    "fi": "finnish",
    "fr": "french",
    "de": "german",
    "el": "greek",
    "it": "italian",
    "no": "norwegian",
    "pl": "polish",
    "pt": "portuguese",
    "ru": "russian",
    "sl": "slovenian",
    "es": "spanish",
    "sv": "swedish",
    "tr": "turkish",
}

# Maps ISO 639-1 code → spaCy model name.
# "__default__" is the multilingual fallback when no dedicated model exists.
SPACY_MODEL_MAP: dict[str, str] = {
    "en": "en_core_web_lg",
    "fr": "fr_core_news_lg",
    "de": "de_core_news_lg",
    "es": "es_core_news_lg",
    "it": "it_core_news_lg",
    "pt": "pt_core_news_lg",
    "zh": "zh_core_web_sm",
    "ja": "ja_core_news_lg",
    "ru": "ru_core_news_lg",
    "__default__": "xx_core_web_sm",
}

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ru": "Russian",
    "ar": "Arabic",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "el": "Greek",
}


def detect_language(text: str, fallback: str = "en") -> str:
    """Return ISO 639-1 code for the dominant language of *text*.

    Falls back to *fallback* when text is too short, langdetect is not
    installed, or detection throws an exception.
    """
    if not _LANGDETECT_AVAILABLE or len(text.strip()) < 20:
        return fallback
    try:
        return detect(text)
    except Exception:
        return fallback


def language_name(code: str) -> str:
    """Map ISO 639-1 code to an English display name, e.g. "fr" → "French"."""
    return LANGUAGE_NAMES.get(code, "English")


def nltk_language(code: str) -> str:
    """Map ISO 639-1 code to the NLTK punkt tokenizer language name."""
    return NLTK_LANGUAGE_MAP.get(code, "english")


def spacy_model_for(code: str) -> str:
    """Return the preferred spaCy model name for a given language code."""
    return SPACY_MODEL_MAP.get(code, SPACY_MODEL_MAP["__default__"])
