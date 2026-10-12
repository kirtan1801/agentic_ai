"""Layer 1: cheap rule-based checks plus OpenAI's free moderation endpoint."""
import re

from openai import AsyncOpenAI

from app.config import settings

INJECTION_PATTERNS = re.compile(
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|rules|prompts?)"
    r"|disregard (all |any )?(the )?(previous|prior|above|your) (instructions|rules)"
    r"|(reveal|show|print|repeat|leak) (me )?(your |the )?(system|hidden|initial) (prompt|instructions)"
    r"|you are now (a|an|in) "
    r"|pretend (you are|to be) "
    r"|developer mode|jailbreak|\bDAN\b",
    re.IGNORECASE,
)

# Order matters: longer digit patterns first so a card number isn't half-matched as a phone.
PII_PATTERNS = [
    ("[EMAIL]", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("[CARD]", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("[ID]", re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b")),  # Aadhaar-style
    ("[ID]", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),  # SSN-style
    ("[PHONE]", re.compile(r"(?<!\w)\+?\d[\d -]{8,13}\d\b")),
]

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def validate_input(message: str) -> str:
    """Return a cleaned message, or raise ValueError with the reason it was rejected."""
    message = CONTROL_CHARS.sub("", message).strip()

    if not message:
        raise ValueError("Message is empty.")

    if len(message) > settings.max_input_chars:
        raise ValueError(f"Message is too long (max {settings.max_input_chars} characters).")

    if INJECTION_PATTERNS.search(message):
        raise ValueError("Message looks like a prompt-injection attempt.")

    return message


def redact_pii(text: str) -> str:
    for placeholder, pattern in PII_PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


_client: AsyncOpenAI | None = None


async def is_flagged(text: str) -> bool:
    global _client
    _client = _client or AsyncOpenAI(api_key=settings.llm_api_key)

    result = await _client.moderations.create(model="omni-moderation-latest", input=text)
    return result.results[0].flagged
