import logging
import time
from collections import defaultdict, deque

import openai
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

from app.api.routes.chat import router as chat_router
from app.api.routes.docs import router as docs_router
from app.config import settings
from app.core.exception import GuardrailError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# Identical LLM prompts are answered from memory instead of re-billed.
# ponytail: per-process cache, swap to SQLiteCache/Redis when running multiple workers.
set_llm_cache(InMemoryCache(maxsize=1000))

app = FastAPI(
    title="AI Agent",
    version="0.1.0",
    # /docs is the documents API; serve Swagger UI elsewhere.
    docs_url="/swagger",
)

# ponytail: in-memory per-IP limiter, single process only; move to Redis behind a proxy.
_hits: dict[str, deque[float]] = defaultdict(deque)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.method == "POST":
        ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        hits = _hits[ip]
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= settings.rate_limit_per_minute:
            return JSONResponse(status_code=429, content={"detail": "Too many requests. Try again in a minute."})
        hits.append(now)
    return await call_next(request)


@app.exception_handler(GuardrailError)
async def guardrail_error_handler(request: Request, exc: GuardrailError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# Moderation runs outside the route's try/except; if OpenAI is unreachable,
# reject the request cleanly (fail closed) instead of a raw 500 traceback.
@app.exception_handler(openai.APIError)
async def openai_error_handler(request: Request, exc: openai.APIError):
    logging.getLogger(__name__).error("OpenAI call failed: %s", exc)
    return JSONResponse(status_code=503, content={"detail": "AI service is unreachable right now. Please try again."})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

app.include_router(chat_router)
app.include_router(docs_router)
