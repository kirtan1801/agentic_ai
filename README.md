# AI Agent (LangChain + LangGraph)

FastAPI backend with a LangGraph tool-calling agent (document search + web search), Pinecone RAG with reranking and citations, streaming responses, and a Vite + React UI.

## Run

```bash
uv run uvicorn app.main:app --reload     # API on :8000 (Swagger UI at /swagger)
cd frontend && npm install && npm run dev  # UI on :5173 (proxies /api -> :8000)
```

## Settings (`.env`)

Required: `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, `TAVILY_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`.

Tracing (optional): `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`. Runs are named `chat-<mode>` and grouped by `thread_id`.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_ROUTER_MODEL` | `LLM_MODEL` | Cheaper model for routing + guardrail checks |
| `LLM_MAX_TOKENS` | 3000 | Output token cap |
| `RETRIEVAL_MIN_SCORE` | 0.35 | Drop weak RAG matches (when reranking is off or fails) |
| `RERANK_ENABLED` | true | Pinecone hosted reranker on retrieved chunks |
| `RERANK_MODEL` | `bge-reranker-v2-m3` | Reranker model |
| `RERANK_MIN_SCORE` | 0.05 | Drop chunks below this rerank score |
| `MAX_TOOL_CALLS` | 4 | Agent tool calls per turn before it must answer |
| `MAX_ANSWER_RETRIES` | 1 | Regenerations of an answer not supported by its sources (0 = off) |
| `MAX_UPLOAD_BYTES` | 5 MB | Upload size cap |
| `RATE_LIMIT_PER_MINUTE` | 20 | POST requests per IP |
| `MAX_INPUT_CHARS` | 2000 | Message length cap |
| `GUARD_MODERATION` | true | OpenAI moderation on input + output |
| `GUARD_NEMO` | false | NeMo Guardrails self-check rails |
| `GUARD_LLM_JUDGE` | false | LLM judge for safety + groundedness |

## Guardrails

`app/guardrails/`: every chat route calls `check_input` then `check_output`.
Rule checks (length, prompt-injection patterns, PII redaction) always run. Retrieved docs and web results are wrapped in `<untrusted>` tags.

Test: `uv run python -m tests.test_guardrails`

## Agent

- **Agent** mode: tool-calling loop (`search_documents`, `web_search`), capped at `MAX_TOOL_CALLS`, plus a hard graph `recursion_limit`.
- **Ask docs** mode: follow-up rewrite → retrieval (top 12 → rerank to 4) → cited answer.
- **Chat** mode: plain LLM with history.
- Agent and Ask answers are checked for groundedness; unsupported answers are regenerated at most `MAX_ANSWER_RETRIES` times.
- `POST /chat/stream` streams server-sent events: `status`, `token`, `reset`, `done`, `error`.

## Documents

Upload `.pdf`, `.md`, `.txt`, `.html` (`POST /docs`; re-upload replaces the old version). `GET /docs` lists documents, `DELETE /docs/{source}` removes one. Chat requests accept `sources: [...]` to search only those files. Answers cite chunks as `[n]` with file and page.

## Conversation memory

All three modes run through one LangGraph graph with a checkpointer (`InMemorySaver`, `app/agent/graph.py`), keyed by `thread_id`.
The UI keeps the id in localStorage; "New chat" starts a new thread. `GET /chat/{thread_id}/history` returns the conversation.
Follow-up questions are rewritten into standalone questions (`contextualize_node`, Ask docs mode) before retrieval; Agent and Chat see recent history directly, trimmed to ~1500 tokens.
History is in memory only: lost on server restart.

Tests (offline, fake models): `uv run python -m tests.test_memory`, `uv run python -m tests.test_agent`
