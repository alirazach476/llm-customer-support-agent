"""Central configuration for the support agent (env + Streamlit secrets)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "knowledge_base.json"
TICKETS_DIR = BASE_DIR / "tickets"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

_PLACEHOLDER_FRAGMENTS = ("your-key", "your_api", "your-api", "****", "sk-your")


def _setting(name: str, default: str = "") -> str:
    """Read Streamlit Cloud secrets first, then environment / .env (fresh each call)."""
    try:
        import streamlit as st

        if name in st.secrets:
            value = st.secrets[name]
            if value is not None and str(value).strip():
                return str(value).strip().strip('"').strip("'")
    except Exception:
        pass
    raw = os.getenv(name, default)
    return (raw or default).strip().strip('"').strip("'")


def llm_provider() -> str:
    return _setting("LLM_PROVIDER", "groq").lower()


def groq_api_key() -> str:
    return _setting("GROQ_API_KEY")


def groq_model() -> str:
    return _setting("GROQ_MODEL", "openai/gpt-oss-20b")


def openai_api_key() -> str:
    return _setting("OPENAI_API_KEY")


def openai_model() -> str:
    return _setting("OPENAI_MODEL", "gpt-4o-mini")


def classify_temperature() -> float:
    return float(_setting("CLASSIFY_TEMPERATURE", "0"))


def reply_temperature() -> float:
    return float(_setting("REPLY_TEMPERATURE", "0.3"))


def escalate_threshold() -> float:
    return float(_setting("ESCALATE_THRESHOLD", "0.45"))


def company_name() -> str:
    return _setting("COMPANY_NAME", "NovaMart")


# Back-compat names used by app.py / tests (resolved at import; prefer functions above for keys).
LLM_PROVIDER = llm_provider()
GROQ_MODEL = groq_model()
OPENAI_MODEL = openai_model()
CLASSIFY_TEMPERATURE = classify_temperature()
REPLY_TEMPERATURE = reply_temperature()
ESCALATE_THRESHOLD = escalate_threshold()
COMPANY_NAME = company_name()
GROQ_API_KEY = groq_api_key()
OPENAI_API_KEY = openai_api_key()


def _looks_like_real_key(key: str, prefix: str) -> bool:
    if not key or not key.startswith(prefix):
        return False
    lower = key.lower()
    return not any(frag in lower for frag in _PLACEHOLDER_FRAGMENTS)


def api_key_is_configured() -> bool:
    """True when the active provider has a real-looking API key."""
    if llm_provider() == "openai":
        return _looks_like_real_key(openai_api_key(), "sk-")
    return _looks_like_real_key(groq_api_key(), "gsk_")


def api_setup_help() -> str:
    if llm_provider() == "openai":
        return (
            "OPENAI_API_KEY is missing. Add it in Streamlit Secrets (cloud) or .env (local)."
        )
    return (
        "GROQ_API_KEY is missing or invalid. "
        "On Streamlit Cloud: App settings → Secrets → set GROQ_API_KEY = \"gsk_...\" "
        "then reboot the app. Free key: https://console.groq.com/keys"
    )


def require_api_key() -> None:
    """Raise a clear error when the active provider key is missing."""
    if not api_key_is_configured():
        raise RuntimeError(api_setup_help())


def llm_model_name() -> str:
    return openai_model() if llm_provider() == "openai" else groq_model()
