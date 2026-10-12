import logging

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.state import AgentState, Mode
from app.agent.graph import graph

logger = logging.getLogger(__name__)


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


class AgentService:
    async def run(self, thread_id: str, mode: Mode, message: str) -> dict:
        """Run one turn of the conversation. Returns the response, sources and grounding context."""
        logger.info("Agent request received: thread=%s mode=%s", thread_id, mode)

        # Per-turn fields are reset explicitly so the previous turn's values don't leak in.
        state: AgentState = {
            'messages': [HumanMessage(message)],
            'mode': mode,
            'message': message,
            'decision': None,
            'response': None,
            'tool_result': None,
            'rag_result': None,
            'sources': None,
        }

        result = await graph.ainvoke(state, _config(thread_id))

        logger.info("Agent graph completed")

        response = result['response']

        if response is None: 
            logger.error("Agent response missing after graph execution")
            raise ValueError("Agent response is missing.")

        context = "\n\n".join(
            part for part in (result["rag_result"], result["tool_result"]) if part
        )

        return {"response": response, "sources": result["sources"] or [], "context": context}

    async def save_answer(self, thread_id: str, answer: str) -> None:
        # Called after output guardrails, so blocked answers never enter the history.
        await graph.aupdate_state(_config(thread_id), {"messages": [AIMessage(answer)]})

    async def history(self, thread_id: str) -> list[dict]:
        snapshot = await graph.aget_state(_config(thread_id))
        return [
            {"role": "user" if m.type == "human" else "assistant", "text": m.content}
            for m in snapshot.values.get("messages", [])
        ]
