"""LLM Customer Support Agent with intent routing and knowledge-base replies."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

BASE = Path(__file__).resolve().parent
KB_PATH = BASE / "knowledge_base.json"
TICKETS_DIR = BASE / "tickets"


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


def load_kb() -> dict:
    return json.loads(KB_PATH.read_text(encoding="utf-8"))


def get_llm(temp: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=temp)


def classify_intent(state: SupportState) -> dict:
    llm = get_llm(0).with_structured_output(IntentResult)
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
    return {
        "intent": result.intent,
        "confidence": result.confidence,
        "summary": result.summary,
        "escalate": result.intent == "escalate" or result.confidence < 0.45,
    }


def draft_reply(state: SupportState) -> dict:
    if state.get("escalate"):
        reply = (
            "Thanks for reaching out. I'm connecting you with a human specialist "
            f"for this {state.get('intent', 'sensitive')} issue. "
            "Your ticket has been prioritized."
        )
        return {"reply": reply}

    kb = load_kb()
    section = kb.get(state["intent"], kb.get("general", {}))
    llm = get_llm(0.3)
    prompt = (
        f"You are a helpful customer support agent for {kb.get('company', 'our company')}.\n"
        f"Intent: {state['intent']}\n"
        f"Issue summary: {state['summary']}\n"
        f"Knowledge base facts:\n{json.dumps(section, indent=2)}\n\n"
        f"Customer message:\n{state['customer_message']}\n\n"
        "Write a concise, empathetic reply. Do not invent policies."
    )
    reply = llm.invoke([HumanMessage(content=prompt)]).content
    return {"reply": reply}


def create_ticket(state: SupportState) -> dict:
    TICKETS_DIR.mkdir(parents=True, exist_ok=True)
    ticket_id = datetime.now(timezone.utc).strftime("T%Y%m%d%H%M%S")
    payload = {
        "ticket_id": ticket_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "customer_message": state["customer_message"],
        "intent": state["intent"],
        "confidence": state["confidence"],
        "summary": state["summary"],
        "escalate": state["escalate"],
        "reply": state["reply"],
    }
    (TICKETS_DIR / f"{ticket_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"ticket_id": ticket_id}


def build_support_graph():
    graph = StateGraph(SupportState)
    graph.add_node("classify", classify_intent)
    graph.add_node("reply", draft_reply)
    graph.add_node("ticket", create_ticket)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "reply")
    graph.add_edge("reply", "ticket")
    graph.add_edge("ticket", END)
    return graph.compile()


def handle_message(customer_message: str) -> SupportState:
    app = build_support_graph()
    return app.invoke(
        {
            "customer_message": customer_message,
            "intent": "",
            "confidence": 0.0,
            "summary": "",
            "reply": "",
            "escalate": False,
            "ticket_id": "",
        }
    )


def main() -> None:
    print("LLM Customer Support Agent")
    print("Type a customer message (or 'quit').\n")
    while True:
        msg = input("Customer> ").strip()
        if not msg:
            continue
        if msg.lower() in {"quit", "exit", "q"}:
            break
        result = handle_message(msg)
        print(f"\nIntent: {result['intent']} ({result['confidence']:.2f})")
        print(f"Escalate: {result['escalate']} | Ticket: {result['ticket_id']}")
        print(f"Agent> {result['reply']}\n")


if __name__ == "__main__":
    main()
