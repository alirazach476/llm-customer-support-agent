"""NovaMart support console — fast Streamlit dashboard."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from config import (
    COMPANY_NAME,
    ESCALATE_THRESHOLD,
    LLM_PROVIDER,
    api_key_is_configured,
    api_setup_help,
    llm_model_name,
)

st.set_page_config(
    page_title=f"{COMPANY_NAME} Support Console",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

EXAMPLE_MESSAGES = [
    ("Billing", "I was charged twice for order #4521. Can I get a refund?"),
    ("Shipping", "My package still hasn't arrived and it's been a week."),
    ("Technical", "The app freezes on the checkout spinner every time."),
    ("Account", "I need to change the email on my account."),
    ("Escalate", "This is unacceptable — I'm contacting my lawyer about this refund."),
]


@st.cache_resource(show_spinner=False)
def _warm_agent():
    """Preload KB + compiled LangGraph once (makes later runs feel instant)."""
    from support_agent import build_support_graph, load_kb

    load_kb()
    return build_support_graph()


def _run_ticket(message: str) -> dict[str, Any]:
    _warm_agent()
    from support_agent import handle_message

    return handle_message(message)


DASHBOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Syne:wght@600;700;800&display=swap');

html, body, [class*="css"]  {
  font-family: 'Manrope', sans-serif;
}

.stApp {
  background:
    radial-gradient(1200px 600px at 10% -10%, #c8ebe3 0%, transparent 55%),
    radial-gradient(900px 500px at 100% 0%, #d7e4f5 0%, transparent 50%),
    linear-gradient(180deg, #f3f6f5 0%, #e9efed 100%);
}

.block-container {
  padding-top: 1.1rem !important;
  padding-bottom: 2rem !important;
  max-width: 1180px;
}

[data-testid="stSidebar"] {
  background: #0b1f1c;
}
[data-testid="stSidebar"] * { color: #e8f2ef !important; }
[data-testid="stSidebar"] .stCaption { opacity: 0.75; }

.brand-bar {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1.25rem;
  animation: rise 420ms ease-out;
}
.brand-bar h1 {
  font-family: 'Syne', sans-serif;
  font-weight: 800;
  font-size: clamp(1.8rem, 3vw, 2.45rem);
  letter-spacing: -0.03em;
  color: #0b1f1c;
  margin: 0;
  line-height: 1.05;
}
.brand-bar p {
  margin: 0.35rem 0 0;
  color: #3d5a54;
  font-size: 0.98rem;
}
.pill {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  background: #0b1f1c;
  color: #d7f5ee;
  border-radius: 999px;
  padding: 0.45rem 0.85rem;
  font-size: 0.78rem;
  font-weight: 600;
  white-space: nowrap;
}
.pill span { opacity: 0.65; font-weight: 500; }

.panel {
  background: rgba(255,255,255,0.82);
  border: 1px solid rgba(15, 61, 52, 0.08);
  border-radius: 18px;
  padding: 1.15rem 1.25rem;
  box-shadow: 0 10px 30px rgba(11, 31, 28, 0.05);
  backdrop-filter: blur(8px);
  animation: rise 480ms ease-out;
}
.panel h3 {
  font-family: 'Syne', sans-serif;
  font-size: 1.05rem;
  margin: 0 0 0.85rem;
  color: #0b1f1c;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.75rem;
  margin: 0.85rem 0 1rem;
  animation: rise 520ms ease-out;
}
.kpi {
  background: #ffffff;
  border: 1px solid rgba(15, 61, 52, 0.08);
  border-radius: 16px;
  padding: 0.95rem 1rem;
  transition: transform 180ms ease, box-shadow 180ms ease;
}
.kpi:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 24px rgba(11, 31, 28, 0.08);
}
.kpi .label {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #5b746e;
  font-weight: 600;
}
.kpi .value {
  font-family: 'Syne', sans-serif;
  font-size: 1.35rem;
  font-weight: 700;
  color: #0b1f1c;
  margin-top: 0.25rem;
  word-break: break-all;
}
.kpi.alert { border-color: #f0b4b4; background: #fff7f7; }
.kpi.alert .value { color: #b42318; }
.kpi.ok { border-color: #b7e0d4; background: #f3fbf8; }
.kpi.ok .value { color: #0f766e; }

.meter {
  height: 8px;
  border-radius: 999px;
  background: #d9e7e2;
  overflow: hidden;
  margin: 0.2rem 0 1rem;
}
.meter > i {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #0f766e, #14b8a6);
}

.reply-box {
  background: #f7fbfa;
  border: 1px solid #d5e8e2;
  border-left: 4px solid #0f766e;
  color: #0b1f1c !important;
  padding: 1rem 1.15rem;
  border-radius: 0 14px 14px 0;
  line-height: 1.55;
  font-size: 1.02rem;
}
.escalate-box {
  background: #fff5f5;
  border: 1px solid #f0c2c2;
  border-left: 4px solid #c53030;
  color: #7f1d1d !important;
  padding: 0.85rem 1.05rem;
  border-radius: 0 14px 14px 0;
  margin-bottom: 0.85rem;
  font-weight: 600;
}

.empty-state {
  text-align: center;
  padding: 2.2rem 1rem;
  color: #4b655f;
}
.empty-state strong {
  display: block;
  font-family: 'Syne', sans-serif;
  font-size: 1.2rem;
  color: #0b1f1c;
  margin-bottom: 0.35rem;
}

div.stButton > button[kind="primary"] {
  background: #0f766e !important;
  border: none !important;
  color: white !important;
  font-weight: 700 !important;
  border-radius: 12px !important;
  height: 2.8rem;
  transition: transform 150ms ease, filter 150ms ease;
}
div.stButton > button[kind="primary"]:hover {
  filter: brightness(1.06);
  transform: translateY(-1px);
}
div.stButton > button[kind="secondary"] {
  border-radius: 10px !important;
  border: 1px solid #c9ddd6 !important;
  background: #ffffff !important;
  color: #0b1f1c !important;
}

@keyframes rise {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 900px) {
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
  .brand-bar { flex-direction: column; align-items: flex-start; }
}
</style>
"""

st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "message_box" not in st.session_state:
    st.session_state.message_box = ""

with st.sidebar:
    st.markdown(f"### {COMPANY_NAME}")
    st.caption("Support operations console")
    st.caption(f"{LLM_PROVIDER} · {llm_model_name()}")
    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown("Classify → Ground → Ticket → Escalate")
    st.caption(f"Escalate under {ESCALATE_THRESHOLD:.0%} confidence")
    st.markdown("---")
    if st.button("Clear session", use_container_width=True):
        st.session_state.history = []
        st.session_state.message_box = ""
        st.rerun()

st.markdown(
    f"""
    <div class="brand-bar">
      <div>
        <h1>{html.escape(COMPANY_NAME)}</h1>
        <p>Support console — grounded replies, logged tickets, smart escalation</p>
      </div>
      <div class="pill">Live <span>· {html.escape(LLM_PROVIDER)}</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not api_key_is_configured():
    st.error(api_setup_help())
    st.info(
        "Free setup: get a key at [console.groq.com/keys](https://console.groq.com/keys), "
        "add `GROQ_API_KEY` to `.env`, restart Streamlit."
    )
    st.stop()

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown('<div class="panel"><h3>New ticket</h3></div>', unsafe_allow_html=True)
    msg = st.text_area(
        "Customer message",
        height=150,
        placeholder="Describe the customer issue…",
        label_visibility="collapsed",
        key="message_box",
    )

    b1, b2 = st.columns([1, 1.4])
    with b1:
        run = st.button("Handle ticket", type="primary", use_container_width=True)
    with b2:
        st.caption("Answers stay grounded in NovaMart policy facts.")

    st.markdown("**Quick scenarios**")
    cols = st.columns(5)
    for i, (label, text) in enumerate(EXAMPLE_MESSAGES):
        with cols[i]:
            if st.button(label, key=f"ex_{i}", use_container_width=True):
                st.session_state.message_box = text
                st.rerun()

with right:
    st.markdown('<div class="panel"><h3>Console status</h3></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Session tickets", len(st.session_state.history))
    c2.metric("Provider", LLM_PROVIDER.upper())
    c3.metric("Threshold", f"{ESCALATE_THRESHOLD:.0%}")
    st.caption("Graph & knowledge base are cached for fast repeat runs.")

if run:
    if not msg.strip():
        st.warning("Enter a customer message first.")
    else:
        with st.spinner("Routing intent and drafting a grounded reply…"):
            try:
                result = _run_ticket(msg.strip())
                st.session_state.history.insert(0, result)
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
                st.stop()

if st.session_state.history:
    latest = st.session_state.history[0]
    conf = min(max(float(latest["confidence"]), 0.0), 1.0)
    escalate = bool(latest["escalate"])
    esc_class = "alert" if escalate else "ok"
    esc_label = "Yes" if escalate else "No"
    ticket = html.escape(str(latest["ticket_id"]))
    intent = html.escape(str(latest["intent"]))
    reply = html.escape(str(latest["reply"])).replace("\n", "<br>")

    st.markdown(
        f"""
        <div class="kpi-grid">
          <div class="kpi"><div class="label">Intent</div><div class="value">{intent}</div></div>
          <div class="kpi"><div class="label">Confidence</div><div class="value">{conf:.0%}</div></div>
          <div class="kpi {esc_class}"><div class="label">Escalated</div><div class="value">{esc_label}</div></div>
          <div class="kpi"><div class="label">Ticket</div><div class="value" style="font-size:1rem">{ticket}</div></div>
        </div>
        <div class="meter"><i style="width:{conf*100:.0f}%"></i></div>
        """,
        unsafe_allow_html=True,
    )

    if escalate:
        st.markdown(
            '<div class="escalate-box">Escalated to human support — '
            "low confidence or a sensitive intent triggered handoff.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="panel"><h3>Agent reply</h3><div class="reply-box">{reply}</div></div>',
        unsafe_allow_html=True,
    )

    with st.expander("Ticket details", expanded=False):
        st.json(
            {
                "ticket_id": latest["ticket_id"],
                "intent": latest["intent"],
                "confidence": latest["confidence"],
                "summary": latest["summary"],
                "escalate": latest["escalate"],
                "customer_message": latest["customer_message"],
            }
        )

    if len(st.session_state.history) > 1:
        st.markdown("#### Session history")
        for item in st.session_state.history[1:6]:
            label = f"{item['ticket_id']} · {item['intent']} · {item['confidence']:.0%}"
            with st.expander(label):
                st.write(item["reply"])
else:
    st.markdown(
        """
        <div class="panel empty-state">
          <strong>Ready when you are</strong>
          Pick a quick scenario or paste a customer message, then hit Handle ticket.
        </div>
        """,
        unsafe_allow_html=True,
    )
