import logging

from fastapi import APIRouter, HTTPException
from langchain_core.callbacks import get_usage_metadata_callback

from app.guardrails import check_input, check_output
from app.schema.chat import ChatRequest, ChatResponse
from app.service.chat import ChatService
from app.service.agent import AgentService
from app.rag.chain import answer_question

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["V1"])

chat_service = ChatService()
agent_service = AgentService()


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    message = await check_input(request.message)
    try:
        with get_usage_metadata_callback() as usage:
            response = await chat_service.chat(message)
        logger.info("Token usage /chat: %s", usage.usage_metadata)
    except Exception as exc:
        logger.exception("Failed to process chat request")
        raise HTTPException(status_code=500, detail="Failed to process the request.") from exc

    return ChatResponse(response=await check_output(message, response))
    
@router.post("/decide", response_model=ChatResponse)
async def decide(request: ChatRequest) -> ChatResponse:
    message = await check_input(request.message)
    try:
        with get_usage_metadata_callback() as usage:
            result = await agent_service.decide(message)
        logger.info("Token usage /chat/decide: %s", usage.usage_metadata)
    except Exception as exc:
        logger.exception("Failed to process agent request")
        raise HTTPException(status_code=500, detail="Failed to process the request.") from exc

    return ChatResponse(
        response=await check_output(message, result["response"], result["context"])
    )
    
@router.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest) -> ChatResponse:
    message = await check_input(request.message)
    try:
        with get_usage_metadata_callback() as usage:
            result = await answer_question(message)
        logger.info("Token usage /chat/ask: %s", usage.usage_metadata)
    except Exception as exc:
        logger.exception("Failed to process ask request")
        raise HTTPException(status_code=500, detail="Failed to generate an answer.") from exc

    return ChatResponse(
        response=await check_output(message, result["answer"], result["context"]),
        sources=result["sources"],
    )
