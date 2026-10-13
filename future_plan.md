# Future plan

Work deferred from the current iteration, in suggested order. Each item lists where it goes in the code.

## 1. Reliability

- **Retries per node**: add `retry_policy=RetryPolicy(max_attempts=3, retry_on=...)` to `builder.add_node(...)` in `app/agent/graph.py` for nodes that call external services (`agent`, `tools`, `rag`). Retry only on transient errors (timeouts, 429, 5xx), never on validation errors.
- **Timeouts**: pass `timeout=` to `init_chat_model` in `app/ai/llm.py` and to Tavily; wrap Pinecone calls with `asyncio.wait_for`. Today a hung upstream hangs the request.
- **Model fallback**: `llm.with_fallbacks([backup_llm])` in `app/ai/llm.py` (e.g. a second provider/model), so one provider outage doesn't take the app down.
- **Graceful tool failures**: `ToolNode(handle_tool_errors=...)` already returns errors to the model; add a user-facing message when web search is down.

## 2. Memory & cost

Done: SQLite history (`data/agent.sqlite`, attached in the `app/main.py` lifespan), running summary of older turns (`summarize_node`), long-term memory per browser `user_id` (`remember` tool + SQLite `Store`), per-thread token budget (`MAX_THREAD_TOKENS`, 429 past it).

- **Thread cleanup**: old threads are never evicted; add a job using `checkpointer.adelete_thread` once auth gives threads an owner.
- **Multiple workers**: SQLite is single-machine; switch to `PostgresSaver` / `PostgresStore` (`langgraph-checkpoint-postgres`).
- **Memory dedupe**: facts are only appended; merge or edit them when the 20-fact prompt cap starts to bite.
- **Shared cache**: skipped for now. `InMemoryCache` only hits byte-identical prompts, and chat prompts include history, so hits are rare. Measure the hit rate in LangSmith first; then Redis/SQLite or a semantic cache.

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
