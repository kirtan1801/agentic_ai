"""Guardrail entry points. Every chat route goes through check_input / check_output.

Layers (toggle in .env):
  1. basic rules + OpenAI moderation  (GUARD_MODERATION, rules always on)
  2. NeMo Guardrails self-check rails (GUARD_NEMO)
  3. LLM-as-judge on the answer       (GUARD_LLM_JUDGE)
"""
import logging

from app.config import settings
from app.core.exception import GuardrailError
from app.guardrails.basic import is_flagged, redact_pii, validate_input

logger = logging.getLogger(__name__)

BLOCKED_ANSWER = "Sorry, I can't share that response."


async def check_input(message: str) -> str:
    try:
        message = validate_input(message)
    except ValueError as exc:
        raise GuardrailError(str(exc)) from exc

    message = redact_pii(message)

    if settings.guard_moderation and await is_flagged(message):
        raise GuardrailError("Message violates the content policy.")

    if settings.guard_nemo:
        from app.guardrails.nemo_rails import nemo_blocks_input

        if await nemo_blocks_input(message):
            raise GuardrailError("Message blocked by input rails.")

    return message


async def check_output(message: str, answer: str, context: str = "") -> str:
    answer = redact_pii(answer)

    if settings.guard_moderation and await is_flagged(answer):
        logger.warning("Answer blocked by moderation")
        return BLOCKED_ANSWER

    if settings.guard_nemo:
        from app.guardrails.nemo_rails import nemo_blocks_output

        if await nemo_blocks_output(message, answer):
            logger.warning("Answer blocked by NeMo output rails")
            return BLOCKED_ANSWER

    if settings.guard_llm_judge:
        from app.guardrails.judge import judge_answer

        verdict = await judge_answer(message, answer, context)
        if not verdict.safe:
            logger.warning("Answer blocked by judge: %s", verdict.reason)
            return BLOCKED_ANSWER
        if not verdict.grounded:
            logger.warning("Answer not grounded: %s", verdict.reason)
            answer += "\n\n_Note: parts of this answer may not be supported by the sources._"

    return answer
