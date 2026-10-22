"""Streamlit UI for the LLM customer support agent."""

import streamlit as st

from support_agent import handle_message

st.set_page_config(page_title="Support Agent", page_icon="🎧", layout="centered")
st.title("LLM Customer Support Agent")
st.caption("Intent classification → grounded reply → ticket logging")

msg = st.text_area("Customer message", height=120, placeholder="I was charged twice for order #4521...")
if st.button("Handle ticket", type="primary") and msg.strip():
    with st.spinner("Routing and drafting reply..."):
        result = handle_message(msg.strip())
    st.metric("Intent", result["intent"])
    st.write(f"Confidence: {result['confidence']:.2f}")
    st.write(f"Escalate: {'Yes' if result['escalate'] else 'No'}")
    st.write(f"Ticket ID: `{result['ticket_id']}`")
    st.subheader("Agent reply")
    st.write(result["reply"])
