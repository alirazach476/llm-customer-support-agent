"""Central configuration for the support agent (env + Streamlit secrets)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "knowledge_base.json"
TICKETS_DIR = BASE_DIR / "tickets"


def _setting(name: str, default: str = "") -> str:
    """Read Streamlit Cloud secrets first, then environment / .env."""
    try:
        import streamlit as st

        if name in st.secrets:
            value = st.secrets[name]
            if value is not None and str(value).strip():
                return str(value).strip()
    except Exception:
        pass
    return os.getenv(name, default).strip() or default


# Default provider is Groq (free tier). Set LLM_PROVIDER=openai to use OpenAI instead.
LLM_PROVIDER = _setting("LLM_PROVIDER", "groq").lower()

GROQ_API_KEY = _setting("GROQ_API_KEY")
GROQ_MODEL = _setting("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

OPENAI_API_KEY = _setting("OPENAI_API_KEY")
OPENAI_MODEL = _setting("OPENAI_MODEL", "gpt-4o-mini")

CLASSIFY_TEMPERATURE = float(_setting("CLASSIFY_TEMPERATURE", "0"))
REPLY_TEMPERATURE = float(_setting("REPLY_TEMPERATURE", "0.3"))
ESCALATE_THRESHOLD = float(_setting("ESCALATE_THRESHOLD", "0.45"))
COMPANY_NAME = _setting("COMPANY_NAME", "NovaMart")

_PLACEHOLDER_FRAGMENTS = ("your-key", "your_api", "your-api", "****", "sk-your")


def _looks_like_real_key(key: str, prefix: str) -> bool:
    if not key or not key.startswith(prefix):
        return False
    lower = key.lower()
    return not any(frag in lower for frag in _PLACEHOLDER_FRAGMENTS)


def api_key_is_configured() -> bool:
    """True when the active provider has a real-looking API key."""
    if LLM_PROVIDER == "openai":
        return _looks_like_real_key(OPENAI_API_KEY, "sk-")
    return _looks_like_real_key(GROQ_API_KEY, "gsk_")


def api_setup_help() -> str:
    if LLM_PROVIDER == "openai":
        return (
            "OPENAI_API_KEY is missing. Add it in Streamlit Secrets (cloud) or .env (local)."
        )
    return (
        "GROQ_API_KEY is missing. For local: put it in .env. "
        "For Streamlit Cloud: App settings → Secrets → "
        'GROQ_API_KEY = "gsk_..." '
        "(free key: https://console.groq.com/keys)."
    )


def require_api_key() -> None:
    """Raise a clear error when the active provider key is missing."""
    if not api_key_is_configured():
        raise RuntimeError(api_setup_help())


def llm_model_name() -> str:
    return OPENAI_MODEL if LLM_PROVIDER == "openai" else GROQ_MODEL
