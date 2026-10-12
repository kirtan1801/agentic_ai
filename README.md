# AI Agent (LangChain + LangGraph)

FastAPI backend with a LangGraph agent (direct / RAG / web search / combined), Pinecone RAG, and a Vite + React UI.

## Run

```bash
uv run uvicorn app.main:app --reload     # API on :8000
cd frontend && npm install && npm run dev  # UI on :5173 (proxies /api -> :8000)
```

## Settings (`.env`)

Required: `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, `TAVILY_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_ROUTER_MODEL` | `LLM_MODEL` | Cheaper model for routing + guardrail checks |
| `LLM_MAX_TOKENS` | 3000 | Output token cap |
| `RETRIEVAL_MIN_SCORE` | 0.35 | Drop weak RAG matches |
| `MAX_UPLOAD_BYTES` | 2 MB | Upload size cap |
| `RATE_LIMIT_PER_MINUTE` | 20 | POST requests per IP |
| `MAX_INPUT_CHARS` | 2000 | Message length cap |
| `GUARD_MODERATION` | true | OpenAI moderation on input + output |
| `GUARD_NEMO` | false | NeMo Guardrails self-check rails |
| `GUARD_LLM_JUDGE` | false | LLM judge for safety + groundedness |

## Guardrails

`app/guardrails/`: every chat route calls `check_input` then `check_output`.
Rule checks (length, prompt-injection patterns, PII redaction) always run. Retrieved docs and web results are wrapped in `<untrusted>` tags.

Test: `uv run python -m tests.test_guardrails`

## Conversation memory

All three modes run through one LangGraph graph with a checkpointer (`InMemorySaver`, `app/agent/graph.py`), keyed by `thread_id`.
The UI keeps the id in localStorage; "New chat" starts a new thread. `GET /chat/{thread_id}/history` returns the conversation.
Follow-up questions are rewritten into standalone questions (`contextualize_node`) before routing/retrieval; history sent to the LLM is trimmed to ~1500 tokens.
History is in memory only: lost on server restart.

Test: `uv run python -m tests.test_memory`
