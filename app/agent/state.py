from typing import TypedDict

from app.schema.ai import AgentDecision

class AgentState(TypedDict):
    message: str
    decision: AgentDecision | None
    response: str | None
    tool_result: str | None
    rag_result: str | None