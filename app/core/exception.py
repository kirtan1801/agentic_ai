class AppError(Exception):
    """Base application exception"""

class LLMError(Exception):
    """LLM Operation failed"""

class GuardrailError(AppError):
    """Request or response blocked by a guardrail"""
