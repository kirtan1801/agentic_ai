from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy

from app.agent.node import (
    agent_node,
    contextualize_node,
    direct_response_node,
    ground_check_node,
    rag_node,
    summarize_node,
    route_after_agent,
    route_after_check,
    route_by_mode,
)
from app.agent.state import AgentState
from app.agent.tools import TOOLS

builder = StateGraph(AgentState)

builder.add_node("summarize", summarize_node)
builder.add_node("contextualize", contextualize_node)
builder.add_node("direct", direct_response_node)
# Retry a timed-out or dropped Pinecone call. LLM errors aren't retried here: the
# OpenAI client already retries them, and the fallback model takes over after that.
builder.add_node("rag", rag_node, retry_policy=RetryPolicy(max_attempts=2, retry_on=(TimeoutError, ConnectionError)))
builder.add_node("agent", agent_node)
# Tool calls and results live in the per-turn "scratch" list, not the chat history.
builder.add_node("tools", ToolNode(TOOLS, messages_key="scratch"))
builder.add_node("ground_check", ground_check_node)

builder.add_edge(START, "summarize")

# UI mode picks the path: plain chat, docs-only RAG, or the tool-using agent.
builder.add_conditional_edges(
    "summarize",
    route_by_mode,
    {
        "chat": "direct",
        "ask": "contextualize",
        "agent": "agent",
    },
)

builder.add_edge("direct", END)

builder.add_edge("contextualize", "rag")
builder.add_edge("rag", "ground_check")

# Agent loop: agent -> tools -> agent ... until it answers (tool calls capped in agent_node).
builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "ground_check": "ground_check"})
builder.add_edge("tools", "agent")

# Ungrounded answer -> regenerate (bounded by max_answer_retries), else done.
builder.add_conditional_edges(
    "ground_check",
    route_after_check,
    {"ask": "rag", "agent": "agent", "end": END},
)

# The SQLite checkpointer needs a running event loop, so app/main.py attaches it at
# startup (lifespan). Requests only run inside that, so they always have history.
graph = builder.compile()

# Hard backstop on graph steps per turn, on top of the tool/retry limits.
RECURSION_LIMIT = 25
