import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Mode = Literal["agent", "ask", "chat"]

class AgentState(TypedDict):
    # Persisted across turns by the checkpointer: user questions and final answers only.
    messages: Annotated[list[AnyMessage], add_messages]
    # Running summary of messages that fell out of the history window, and how many it covers.
    summary: str | None
    summarized: int
    # Tokens spent by this conversation (each turn adds its usage).
    tokens_used: Annotated[int, operator.add]
    # Per-turn fields, reset by the service on every request.
    user_id: str | None
    scratch: Annotated[list[AnyMessage], add_messages]  # agent tool calls + results
    mode: Mode
    message: str
    sources_filter: list[str] | None
    response: str | None
    context: str | None
    sources: list[dict] | None
    tool_calls: int
    answer_retries: int
    retry: bool
    ground_feedback: str | None
