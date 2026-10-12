import logging
from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from app.agent.state import AgentState, Mode
from app.agent.graph import RECURSION_LIMIT, graph

logger = logging.getLogger(__name__)

TOOL_LABELS = {"search_documents": "Searching documents", "web_search": "Searching the web"}


def _config(thread_id: str, mode: Mode | None = None) -> dict:
    config: dict = {"configurable": {"thread_id": thread_id}}
    if mode:
        # LangSmith: name runs by mode and group them into one thread per conversation.
        config.update(
            run_name=f"chat-{mode}",
            tags=[mode],
            metadata={"thread_id": thread_id, "mode": mode},
            recursion_limit=RECURSION_LIMIT,
        )
    return config


def _input(mode: Mode, message: str, sources: list[str] | None) -> AgentState:
    # Per-turn fields are reset explicitly so the previous turn's values don't leak in.
    return {
        'messages': [HumanMessage(message)],
        'scratch': [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        'mode': mode,
        'message': message,
        'sources_filter': sources,
        'response': None,
        'context': None,
        'sources': None,
        'tool_calls': 0,
        'answer_retries': 0,
        'retry': False,
        'ground_feedback': None,
    }


def _result(values: dict) -> dict:
    if values.get("response") is None:
        logger.error("Agent response missing after graph execution")
        raise ValueError("Agent response is missing.")
    return {
        "response": values["response"],
        "sources": values.get("sources") or [],
        "context": values.get("context") or "",
    }


class AgentService:
    async def run(self, thread_id: str, mode: Mode, message: str, sources: list[str] | None = None) -> dict:
        """Run one turn of the conversation. Returns the response, citations and grounding context."""
        logger.info("Agent request received: thread=%s mode=%s", thread_id, mode)
        result = await graph.ainvoke(_input(mode, message, sources), _config(thread_id, mode))
        logger.info("Agent graph completed")
        return _result(result)

    async def stream(
        self, thread_id: str, mode: Mode, message: str, sources: list[str] | None = None
    ) -> AsyncIterator[tuple[str, object]]:
        """Yield (event, data): status / token / reset while running, then ("final", result)."""
        logger.info("Agent stream received: thread=%s mode=%s", thread_id, mode)
        if mode == "ask":
            yield "status", "Searching documents"

        async for kind, data in graph.astream(
            _input(mode, message, sources),
            _config(thread_id, mode),
            stream_mode=["messages", "updates"],
        ):
            if kind == "messages":
                chunk, meta = data
                if "answer" in (meta.get("tags") or []) and chunk.text:
                    yield "token", chunk.text
                continue

            for node, update in data.items():
                if node == "agent" and update and update.get("scratch"):
                    for call in getattr(update["scratch"][-1], "tool_calls", None) or []:
                        # Text streamed before a tool call is thinking-out-loud: clear it.
                        yield "reset", None
                        query = call["args"].get("query", "")
                        yield "status", f"{TOOL_LABELS.get(call['name'], call['name'])}: {query}"
                elif node == "ground_check" and update and update.get("retry"):
                    yield "reset", None
                    yield "status", "Improving answer (not fully supported by sources)"

        snapshot = await graph.aget_state(_config(thread_id))
        yield "final", _result(snapshot.values)

    async def save_answer(self, thread_id: str, answer: str) -> None:
        # Called after output guardrails, so blocked answers never enter the history.
        await graph.aupdate_state(_config(thread_id), {"messages": [AIMessage(answer)]})

    async def history(self, thread_id: str) -> list[dict]:
        snapshot = await graph.aget_state(_config(thread_id))
        return [
            {"role": "user" if m.type == "human" else "assistant", "text": m.content}
            for m in snapshot.values.get("messages", [])
        ]
