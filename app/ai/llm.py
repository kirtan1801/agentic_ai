from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

from app.config import settings

llm = init_chat_model(
    model=settings.llm_model,
    model_provider=settings.llm_provider,
    api_key=settings.llm_api_key,
    max_tokens=settings.llm_max_tokens,
)

# Cheap model for classification-style calls (routing, guardrail judge).
router_llm = init_chat_model(
    model=settings.llm_router_model or settings.llm_model,
    model_provider=settings.llm_provider,
    api_key=settings.llm_api_key,
    max_tokens=300,
)

# Plain-text answers tagged "answer" so the stream endpoint forwards only these tokens.
# Plain text (not JSON) also means hitting max_tokens truncates instead of failing to parse.
answer_llm = llm.with_config(tags=["answer"]) | StrOutputParser()
