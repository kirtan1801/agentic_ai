import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Path
from langchain_core.callbacks import get_usage_metadata_callback

from app.agent.state import Mode
from app.guardrails import check_input, check_output
from app.schema.chat import ChatRequest, ChatResponse
from app.service.agent import AgentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["V1"])

agent_service = AgentService()


async def run_turn(request: ChatRequest, mode: Mode) -> ChatResponse:
    thread_id = request.thread_id or uuid4().hex
    message = await check_input(request.message)
    try:
        with get_usage_metadata_callback() as usage:
            result = await agent_service.run(thread_id, mode, message)
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

@router.get("/{thread_id}/history")
async def history(thread_id: str = Path(max_length=64, pattern=r"^[\w-]+$")) -> dict:
    return {"thread_id": thread_id, "messages": await agent_service.history(thread_id)}
