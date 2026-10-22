"""Unit tests for the support agent (mocked LLM — no API calls)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import config
from support_agent import (
    IntentResult,
    build_support_graph,
    create_ticket,
    draft_reply,
    handle_message,
    load_kb,
)


@pytest.fixture(autouse=True)
def _clear_caches():
    load_kb.cache_clear()
    build_support_graph.cache_clear()
    yield
    load_kb.cache_clear()
    build_support_graph.cache_clear()


@pytest.fixture
def tickets_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "tickets"
    monkeypatch.setattr(config, "TICKETS_DIR", path)
    import support_agent as sa

    monkeypatch.setattr(sa, "TICKETS_DIR", path)
    return path


def test_load_kb_has_company_and_intents():
    kb = load_kb()
    assert kb["company"] == "NovaMart"
    for key in ("billing", "technical", "shipping", "account", "general"):
        assert key in kb


def test_create_ticket_writes_json(tickets_dir: Path):
    state = {
        "customer_message": "Double charge on order 1",
        "intent": "billing",
        "confidence": 0.91,
        "summary": "Duplicate charge",
        "escalate": False,
        "reply": "We can help with that.",
        "ticket_id": "",
    }
    result = create_ticket(state)
    ticket_id = result["ticket_id"]
    path = tickets_dir / f"{ticket_id}.json"
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["ticket_id"] == ticket_id
    assert payload["intent"] == "billing"
    assert payload["escalate"] is False


def test_draft_reply_escalation_skips_llm():
    state = {
        "customer_message": "I will sue you",
        "intent": "escalate",
        "confidence": 0.95,
        "summary": "Legal threat",
        "escalate": True,
        "reply": "",
        "ticket_id": "",
    }
    with patch("support_agent.get_llm") as mock_llm:
        out = draft_reply(state)
    mock_llm.assert_not_called()
    assert "human specialist" in out["reply"].lower()


def test_draft_reply_uses_kb_section():
    state = {
        "customer_message": "Where is my invoice?",
        "intent": "billing",
        "confidence": 0.88,
        "summary": "Invoice request",
        "escalate": False,
        "reply": "",
        "ticket_id": "",
    }
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Here is how to find your invoice.")
    with patch("support_agent.get_llm", return_value=mock_llm):
        out = draft_reply(state)
    assert "invoice" in out["reply"].lower()
    prompt = mock_llm.invoke.call_args[0][0][0].content
    assert "refund_window_days" in prompt or "invoice_help" in prompt


def test_handle_message_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        handle_message("   ")


def test_handle_message_pipeline(tickets_dir: Path):
    intent = IntentResult(intent="shipping", confidence=0.82, summary="Late package")
    classify_llm = MagicMock()
    classify_llm.with_structured_output.return_value.invoke.return_value = intent

    reply_llm = MagicMock()
    reply_llm.invoke.return_value = MagicMock(
        content="Your order typically arrives in 3-5 business days."
    )

    def fake_get_llm(temperature: float):
        return classify_llm if temperature == 0 else reply_llm

    with (
        patch("support_agent.get_llm", side_effect=fake_get_llm),
        patch("support_agent.CLASSIFY_TEMPERATURE", 0),
    ):
        result = handle_message("My package is late")

    assert result["intent"] == "shipping"
    assert result["escalate"] is False
    assert result["ticket_id"].startswith("T")
    assert "business days" in result["reply"]
    assert (tickets_dir / f"{result['ticket_id']}.json").exists()


def test_low_confidence_forces_escalate(tickets_dir: Path):
    intent = IntentResult(intent="general", confidence=0.2, summary="Unclear")
    classify_llm = MagicMock()
    classify_llm.with_structured_output.return_value.invoke.return_value = intent

    with (
        patch("support_agent.get_llm", return_value=classify_llm),
        patch("support_agent.CLASSIFY_TEMPERATURE", 0),
        patch("support_agent.ESCALATE_THRESHOLD", 0.45),
    ):
        result = handle_message("asdfasdf ???")

    assert result["escalate"] is True
    assert "human specialist" in result["reply"].lower()
