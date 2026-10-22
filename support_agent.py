"""LLM Customer Support Agent with intent routing and knowledge-base replies."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from config import (
    GROQ_BASE_URL,
    KB_PATH,
    TICKETS_DIR,
    classify_temperature,
    company_name,
    escalate_threshold,
    groq_api_key,
    llm_model_name,
    llm_provider,
    openai_api_key,
    reply_temperature,
    require_api_key,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("support_agent")


class IntentResult(BaseModel):
    intent: Literal["billing", "technical", "shipping", "account", "escalate", "general"]
    confidence: float = Field(ge=0, le=1)
    summary: str


class SupportState(TypedDict):
    customer_message: str
    intent: str
    confidence: float
    summary: str
    reply: str
    escalate: bool
    ticket_id: str


@lru_cache(maxsize=1)
def load_kb() -> dict[str, Any]:
    if not KB_PATH.exists():
        raise FileNotFoundError(f"Knowledge base not found: {KB_PATH}")
    return json.loads(KB_PATH.read_text(encoding="utf-8"))


def get_llm(temperature: float) -> ChatOpenAI:
    """Return a chat model. Default is Groq (free); set LLM_PROVIDER=openai for OpenAI."""
    require_api_key()
    model = llm_model_name()
    if llm_provider() == "openai":
        return ChatOpenAI(model=model, temperature=temperature, api_key=openai_api_key())
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=groq_api_key(),
        base_url=GROQ_BASE_URL,
    )


def classify_intent(state: SupportState) -> dict[str, Any]:
    llm = get_llm(classify_temperature()).with_structured_output(IntentResult)
    result: IntentResult = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Classify the customer support message into one intent: "
                    "billing, technical, shipping, account, escalate, general. "
                    "Use escalate for angry/legal/refund threats or unsafe content."
                )
            ),
            HumanMessage(content=state["customer_message"]),
        ]
    )
    escalate = result.intent == "escalate" or result.confidence < escalate_threshold()
    logger.info(
        "Classified intent=%s confidence=%.2f escalate=%s",
        result.intent,
        result.confidence,
        escalate,
    )
    return {
        "intent": result.intent,
        "confidence": result.confidence,
        "summary": result.summary,
        "escalate": escalate,
    }


def draft_reply(state: SupportState) -> dict[str, str]:
    if state.get("escalate"):
        reply = (
            f"Thanks for reaching out to {company_name()}. I'm connecting you with a "
            f"human specialist for this {state.get('intent', 'sensitive')} issue. "
            "Your ticket has been prioritized."
        )
        return {"reply": reply}

    kb = load_kb()
    section = kb.get(state["intent"], kb.get("general", {}))
    company = kb.get("company", company_name())
    llm = get_llm(reply_temperature())
    prompt = (
        f"You are a helpful customer support agent for {company}.\n"
        f"Intent: {state['intent']}\n"
        f"Issue summary: {state['summary']}\n"
        f"Knowledge base facts:\n{json.dumps(section, indent=2)}\n\n"
        f"Customer message:\n{state['customer_message']}\n\n"
        "Write a concise, empathetic reply. Do not invent policies."
    )
    reply = llm.invoke([HumanMessage(content=prompt)]).content
    logger.info("Drafted reply for intent=%s", state["intent"])
    return {"reply": str(reply)}


def create_ticket(state: SupportState) -> dict[str, str]:
    TICKETS_DIR.mkdir(parents=True, exist_ok=True)
    ticket_id = datetime.now(UTC).strftime("T%Y%m%d%H%M%S")
    payload = {
        "ticket_id": ticket_id,
        "created_at": datetime.now(UTC).isoformat(),
        "customer_message": state["customer_message"],
        "intent": state["intent"],
        "confidence": state["confidence"],
        "summary": state["summary"],
        "escalate": state["escalate"],
        "reply": state["reply"],
    }
    (TICKETS_DIR / f"{ticket_id}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    logger.info("Created ticket %s (escalate=%s)", ticket_id, state["escalate"])
    return {"ticket_id": ticket_id}


@lru_cache(maxsize=1)
def build_support_graph() -> Any:
    graph = StateGraph(SupportState)
    # Node names must not collide with SupportState keys (LangGraph constraint).
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("draft_reply", draft_reply)
    graph.add_node("create_ticket", create_ticket)
    graph.add_edge(START, "classify_intent")
    graph.add_edge("classify_intent", "draft_reply")
    graph.add_edge("draft_reply", "create_ticket")
    graph.add_edge("create_ticket", END)
    return graph.compile()


def handle_message(customer_message: str) -> SupportState:
    """Run the full classify → reply → ticket pipeline."""
    message = customer_message.strip()
    if not message:
        raise ValueError("Customer message cannot be empty.")

    app = build_support_graph()
    return app.invoke(
        {
            "customer_message": message,
            "intent": "",
            "confidence": 0.0,
            "summary": "",
            "reply": "",
            "escalate": False,
            "ticket_id": "",
        }
    )


def main() -> None:
    print(f"{company_name()} Support Agent")
    print("Type a customer message (or 'quit').\n")
    while True:
        try:
            msg = input("Customer> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not msg:
            continue
        if msg.lower() in {"quit", "exit", "q"}:
            break
        try:
            result = handle_message(msg)
        except Exception as exc:
            logger.exception("Failed to handle message")
            print(f"Error: {exc}\n")
            continue
        print(f"\nIntent: {result['intent']} ({result['confidence']:.2f})")
        print(f"Escalate: {result['escalate']} | Ticket: {result['ticket_id']}")
        print(f"Agent> {result['reply']}\n")


if __name__ == "__main__":
    main()
