# LLM Customer Support Agent

An LLM support agent that:

1. Classifies intent (`billing`, `technical`, `shipping`, `account`, `escalate`, `general`)
2. Drafts a grounded reply from a JSON knowledge base
3. Logs a support ticket
4. Escalates low-confidence or sensitive cases

Built with **LangChain + LangGraph + OpenAI**.

## Setup
```bash
cd llm-customer-support-agent
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## Run
```bash
python support_agent.py
streamlit run app.py
```

## Files
- `support_agent.py` — agent graph
- `knowledge_base.json` — company policies/facts
- `tickets/` — generated ticket logs

## Sample Outputs
![Pipeline](outputs/pipeline_overview.png)
![Support Output](outputs/sample_support_output.png)
