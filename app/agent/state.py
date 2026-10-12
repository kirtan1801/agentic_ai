from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from app.schema.ai import AgentDecision

Mode = Literal["agent", "ask", "chat"]

class AgentState(TypedDict):
    # Persisted across turns by the checkpointer.
    messages: Annotated[list[AnyMessage], add_messages]
    # Per-turn fields: reset by the service on every request.
    mode: Mode
    message: str
    decision: AgentDecision | None
    response: str | None
    tool_result: str | None
    rag_result: str | None
    sources: list[str] | None
