from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

from app.config import settings


def _model(model: str, max_tokens: int):
    return init_chat_model(
        model=model,
        model_provider=settings.llm_provider,
        api_key=settings.llm_api_key,
        max_tokens=max_tokens,
        timeout=settings.llm_timeout,
    )


def _with_fallback(primary, max_tokens: int):
    # Same provider and key, another model: survives one model's outage, not the provider's.
    # bind_tools / with_structured_output still work on the result and apply to both.
    if not settings.llm_fallback_model:
        return primary
    return primary.with_fallbacks([_model(settings.llm_fallback_model, max_tokens)])


# Plain chat models, for libraries that need a real model object (deepagents, NeMo).
chat_model = _model(settings.llm_model, settings.llm_max_tokens)
router_chat_model = _model(settings.llm_router_model or settings.llm_model, 300)

llm = _with_fallback(chat_model, settings.llm_max_tokens)

# Cheap model for classification-style calls (routing, guardrail judge).
router_llm = _with_fallback(router_chat_model, 300)

# Plain-text answers tagged "answer" so the stream endpoint forwards only these tokens.
# Plain text (not JSON) also means hitting max_tokens truncates instead of failing to parse.
answer_llm = llm.with_config(tags=["answer"]) | StrOutputParser()
