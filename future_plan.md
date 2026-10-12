# Future plan

Work deferred from the current iteration, in suggested order. Each item lists where it goes in the code.

## 1. Reliability

- **Retries per node**: add `retry_policy=RetryPolicy(max_attempts=3, retry_on=...)` to `builder.add_node(...)` in `app/agent/graph.py` for nodes that call external services (`agent`, `tools`, `rag`). Retry only on transient errors (timeouts, 429, 5xx), never on validation errors.
- **Timeouts**: pass `timeout=` to `init_chat_model` in `app/ai/llm.py` and to Tavily; wrap Pinecone calls with `asyncio.wait_for`. Today a hung upstream hangs the request.
- **Model fallback**: `llm.with_fallbacks([backup_llm])` in `app/ai/llm.py` (e.g. a second provider/model), so one provider outage doesn't take the app down.
- **Graceful tool failures**: `ToolNode(handle_tool_errors=...)` already returns errors to the model; add a user-facing message when web search is down.

## 2. Memory & cost

- **Persistent history**: replace `InMemorySaver` in `app/agent/graph.py` with `SqliteSaver` (`uv add langgraph-checkpoint-sqlite`) for a single machine, or `PostgresSaver` (`langgraph-checkpoint-postgres`) for multiple workers. Add a TTL / cleanup job for old threads.
- **Summarize long chats**: instead of only trimming (`recent_history` in `app/agent/node.py`), keep a running summary of older turns in state and prepend it.
- **Long-term memory**: LangGraph `Store` for facts about the user that survive across threads (name, preferences), written by a small "remember" tool.
- **Token budget**: track tokens per thread (from `get_usage_metadata_callback` in `app/api/routes/chat.py`) and refuse or downgrade the model past a limit.
- **Shared cache**: move `InMemoryCache` (`app/main.py`) to Redis/SQLite so it survives restarts and is shared by workers; consider a semantic cache once exact-match hit rate is measured.

## 3. Before real users

- **Auth**: API keys or OAuth/JWT on all routes; scope threads and documents per user (Pinecone namespace or `user_id` metadata filter). Today anyone with network access can read any thread id and delete any document.
- **Shared rate limiting**: move the in-memory limiter in `app/main.py` to Redis; key by user, not IP (all users share one IP behind a proxy).
- **Docker**: `Dockerfile` for the API (uv) + static build of the frontend served by nginx or FastAPI; `docker-compose.yml` with Redis/Postgres.
- **CI**: run `tests/test_*.py`, `npm run build`, and a linter on every push.
- **CORS**: add `CORSMiddleware` if the frontend is served from a different origin than the API.

## Other deferred items

- **Human approval for tools**: LangGraph `interrupt()` before `web_search` in the agent loop; stream an `approval` event, UI shows Approve/Deny, resume with `Command(resume=...)`. Needs a persistent checkpointer to survive restarts.
- **Hybrid (keyword + vector) search**: the current Pinecone index is `cosine`/dense-only. Needs a new index with `dotproduct` metric and sparse vectors (e.g. BM25 encoder), then re-ingest all documents.
- **Evaluations**: build a LangSmith dataset of 20–30 questions with expected routes/answers; run `langsmith.evaluate` with correctness + groundedness evaluators after each change.
- **OCR for scanned PDFs**: `app/rag/loader.py` rejects PDFs with no text layer; add OCR (e.g. `pytesseract` + `pdf2image`, needs system packages) behind a flag.
- **Dependency cleanup**: the stray `pinecone-client` 3.2.2 package in `.venv` is required by nothing and can conflict with `pinecone` 7.x; remove it.
