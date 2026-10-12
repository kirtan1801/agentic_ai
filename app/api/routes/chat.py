import json
import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import StreamingResponse
from langchain_core.callbacks import get_usage_metadata_callback

from app.agent.state import Mode
from app.guardrails import check_input, check_output
from app.schema.chat import ChatRequest, ChatResponse, StreamRequest
from app.service.agent import AgentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["V1"])

agent_service = AgentService()


async def run_turn(request: ChatRequest, mode: Mode) -> ChatResponse:
    thread_id = request.thread_id or uuid4().hex
    message = await check_input(request.message)
    try:
        with get_usage_metadata_callback() as usage:
            result = await agent_service.run(thread_id, mode, message, request.sources)
        logger.info("Token usage mode=%s: %s", mode, usage.usage_metadata)
    except Exception as exc:
        logger.exception("Failed to process %s request", mode)
        raise HTTPException(status_code=500, detail="Failed to process the request.") from exc

    answer = await check_output(message, result["response"], result["context"])
    await agent_service.save_answer(thread_id, answer)

    return ChatResponse(response=answer, sources=result["sources"], thread_id=thread_id)


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    return await run_turn(request, "chat")

@router.post("/decide", response_model=ChatResponse)
async def decide(request: ChatRequest) -> ChatResponse:
    return await run_turn(request, "agent")

@router.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest) -> ChatResponse:
    return await run_turn(request, "ask")


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/stream")
async def stream(request: StreamRequest) -> StreamingResponse:
    """Server-sent events: status / token / reset while running, then done (or error)."""
    thread_id = request.thread_id or uuid4().hex
    # Input guardrails run before streaming starts, so a block is a normal 400.
    message = await check_input(request.message)

    async def events():
        try:
            with get_usage_metadata_callback() as usage:
                async for event, data in agent_service.stream(thread_id, request.mode, message, request.sources):
                    if event != "final":
                        yield _sse(event, data)
                        continue
                    # Output guardrails need the full answer; if they change it,
                    # the client replaces the streamed text with this one.
                    answer = await check_output(message, data["response"], data["context"])
                    await agent_service.save_answer(thread_id, answer)
                    yield _sse("done", {"response": answer, "sources": data["sources"], "thread_id": thread_id})
            logger.info("Token usage mode=%s (stream): %s", request.mode, usage.usage_metadata)
        except Exception:
            logger.exception("Failed to stream %s request", request.mode)
            yield _sse("error", {"detail": "Failed to process the request."})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{thread_id}/history")
async def history(thread_id: str = Path(max_length=64, pattern=r"^[\w-]+$")) -> dict:
    return {"thread_id": thread_id, "messages": await agent_service.history(thread_id)}
