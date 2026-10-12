"""Offline check of conversation memory with fake LLMs. Run: uv run python -m tests.test_memory"""
import os

os.environ["GUARD_MODERATION"] = "false"  # no network calls

from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda

import app.agent.node as node
from app.schema.ai import AgentDecision

seen: dict = {}


def fake_direct(messages):
    seen["direct"] = [m.content for m in messages]
    return SimpleNamespace(answer=f"answer#{len(messages)}")


async def fake_answer_question(question):
    seen["rag_question"] = question
    return {"answer": "rag answer", "sources": ["a.md"], "context": ""}


node.structure_llm = RunnableLambda(fake_direct)
node.router_llm = RunnableLambda(lambda prompt: SimpleNamespace(content="standalone question"))
node.agent_llm = RunnableLambda(
    lambda message: AgentDecision(intent="x", requires_rag=False, requires_tool=False)
)
node.answer_question = fake_answer_question

from app.main import app  # noqa: E402


def test_memory() -> None:
    client = TestClient(app)

    first = client.post("/chat/", json={"message": "My name is Kirtan"}).json()
    thread_id = first["thread_id"]
    assert "direct" in seen and seen["direct"] == ["My name is Kirtan"]

    client.post("/chat/", json={"message": "What's my name?", "thread_id": thread_id})
    assert seen["direct"] == ["My name is Kirtan", "answer#1", "What's my name?"]

    # Shared across modes, and follow-ups are rewritten before retrieval.
    ask = client.post("/chat/ask", json={"message": "its pricing?", "thread_id": thread_id}).json()
    assert ask["sources"] == ["a.md"]
    assert seen["rag_question"] == "standalone question"

    history = client.get(f"/chat/{thread_id}/history").json()["messages"]
    assert [m["role"] for m in history] == ["user", "assistant"] * 3

    assert client.get("/chat/other-thread/history").json()["messages"] == []
    assert client.post("/chat/", json={"message": "hi", "thread_id": "../x"}).status_code == 422


if __name__ == "__main__":
    test_memory()
    print("ok")
