"""Offline check of conversation memory with fake LLMs. Run: uv run python -m tests.test_memory"""
import os

os.environ["GUARD_MODERATION"] = "false"  # no network calls
os.environ["LANGSMITH_TRACING"] = "false"  # don't send test traces
os.environ["CHECKPOINT_DB"] = ":memory:"  # isolated, throwaway history

from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda

import app.agent.node as node
from langchain_core.messages import AIMessage

seen: dict = {}


def fake_direct(messages):
    seen["direct"] = [m.content for m in messages]
    return f"answer#{len(messages)}"


async def fake_answer_question(question, sources=None, feedback=""):
    seen["rag_question"] = question
    citation = {"n": 1, "source": "a.md", "page": 1, "snippet": "s"}
    return {"answer": "rag answer", "sources": [citation], "context": ""}


node.answer_llm = RunnableLambda(fake_direct)
node.router_llm = RunnableLambda(lambda prompt: SimpleNamespace(content="standalone question"))
node.agent_model = lambda: RunnableLambda(lambda messages: AIMessage("agent answer"))
node.answer_question = fake_answer_question

from app.main import app  # noqa: E402


def test_memory() -> None:
    with TestClient(app) as client:
        first = client.post("/chat/", json={"message": "My name is Kirtan"}).json()
        thread_id = first["thread_id"]
        assert "direct" in seen and seen["direct"] == ["My name is Kirtan"]

        client.post("/chat/", json={"message": "What's my name?", "thread_id": thread_id})
        assert seen["direct"] == ["My name is Kirtan", "answer#1", "What's my name?"]

        # Shared across modes, and follow-ups are rewritten before retrieval.
        ask = client.post("/chat/ask", json={"message": "its pricing?", "thread_id": thread_id}).json()
        assert ask["sources"][0]["source"] == "a.md"
        assert seen["rag_question"] == "standalone question"

        history = client.get(f"/chat/{thread_id}/history").json()["messages"]
        assert [m["role"] for m in history] == ["user", "assistant"] * 3

        assert client.get("/chat/other-thread/history").json()["messages"] == []
        assert client.post("/chat/", json={"message": "hi", "thread_id": "../x"}).status_code == 422


def test_history_survives_restart() -> None:
    import asyncio
    import tempfile

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from app.agent.graph import builder
    from app.service.agent import _config

    async def run(path: str) -> None:
        config = _config("restart-thread")
        async with AsyncSqliteSaver.from_conn_string(path) as saver:
            await builder.compile(checkpointer=saver).aupdate_state(config, {"messages": [AIMessage("remembered")]}, as_node="direct")
        # A fresh saver on the same file stands in for a server restart.
        async with AsyncSqliteSaver.from_conn_string(path) as saver:
            snapshot = await builder.compile(checkpointer=saver).aget_state(config)
        assert [m.content for m in snapshot.values["messages"]] == ["remembered"]

    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(run(f"{tmp}/agent.sqlite"))


def test_summary() -> None:
    import asyncio

    from langchain_core.messages import HumanMessage

    long = "word " * 400  # ~500 tokens: only the last few fit the history window
    messages = [(HumanMessage if i % 2 == 0 else AIMessage)(f"{i} {long}", id=str(i)) for i in range(10)]
    state = {"messages": messages, "summary": None, "summarized": 0}

    update = asyncio.run(node.summarize_node(state))
    assert update["summary"] == "standalone question" and update["summarized"] >= node.SUMMARY_BATCH
    # Below the batch size nothing new is summarized (no LLM call).
    assert asyncio.run(node.summarize_node({**state, **update})) == {}

    history = node.recent_history({**state, **update})
    assert history[0].type == "system" and "standalone question" in history[0].content
    assert history[-1] is messages[-1]


def test_long_term_memory() -> None:
    import asyncio

    from langgraph.store.memory import InMemoryStore

    from app.agent.tools import remember

    store = InMemoryStore()

    async def save(user_id):
        return await remember.coroutine(fact="Name is Kirtan", state={"user_id": user_id}, store=store)

    assert asyncio.run(save(None)).startswith("Can't save")
    assert asyncio.run(save("browser-1")) == "Saved."

    # Another chat of the same browser sees the fact; other browsers don't.
    assert "Name is Kirtan" in asyncio.run(node._memories_note({"user_id": "browser-1"}, store))
    assert asyncio.run(node._memories_note({"user_id": "browser-2"}, store)) == ""


def test_token_budget() -> None:
    import app.api.routes.chat as chat_routes
    from app.config import settings

    # Fake LLMs report no usage: pretend every turn costs 5 tokens.
    real_total, real_limit = chat_routes.total_tokens, settings.max_thread_tokens
    chat_routes.total_tokens, settings.max_thread_tokens = (lambda usage: 5), 8
    try:
        with TestClient(app) as client:
            send = lambda: client.post("/chat/", json={"message": "hi", "thread_id": "budget"})
            assert send().status_code == 200  # 0 used
            assert send().status_code == 200  # 5 used
            blocked = send()  # 10 used >= 8
            assert blocked.status_code == 429 and "token limit" in blocked.json()["detail"]
            stream = client.post("/chat/stream", json={"message": "hi", "thread_id": "budget", "mode": "chat"})
            assert stream.status_code == 429
    finally:
        chat_routes.total_tokens, settings.max_thread_tokens = real_total, real_limit


if __name__ == "__main__":
    test_memory()
    test_history_survives_restart()
    test_summary()
    test_long_term_memory()
    test_token_budget()
    print("ok")
