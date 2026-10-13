"""Offline checks for the agent loop, limits, streaming, and PDF loading.
Run: uv run python -m tests.test_agent"""
import os

os.environ["GUARD_MODERATION"] = "false"  # no network calls
os.environ["LANGSMITH_TRACING"] = "false"  # don't send test traces
os.environ["CHECKPOINT_DB"] = ":memory:"  # isolated, throwaway history

import asyncio
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

import app.agent.node as node
import app.agent.tools as tools
import app.rag.chain as chain
import app.tools.web_search as web
from app.config import settings
from app.rag.ingestion import chunk_id, doc_prefix
from app.rag.loader import load_document
from app.schema.ai import GroundCheck

calls = {"agent": 0, "ground": 0, "web": 0, "rag": 0}


async def fake_retrieve(query, k=4, sources=None):
    return [Document(page_content=f"fact about {query}", metadata={"source": "a.pdf", "page": 2})]

tools.retrieve_docs = fake_retrieve
chain.retrieve_docs = fake_retrieve


async def fake_list_documents():
    return [{"source": "a.pdf", "chunks": 3}]


async def fake_read_chunks(source, max_chunks):
    if source != "a.pdf":
        return []
    return [Document(page_content=f"part {i}", metadata={"source": "a.pdf", "page": 1}) for i in range(2)]


node.list_documents = fake_list_documents
tools.read_document_chunks = fake_read_chunks


async def fake_tavily(_):
    calls["web"] += 1
    return {"results": [{"title": "Web", "url": "https://example.com", "content": "web fact"}]}

web.tavily_search = SimpleNamespace(ainvoke=fake_tavily)


def streaming_answer(text):
    model = GenericFakeChatModel(messages=iter([AIMessage(text)] * 20))
    return model.with_config(tags=["answer"]) | StrOutputParser()


def always_call_tools():
    async def reply(messages):
        calls["agent"] += 1
        name = "search_documents" if calls["agent"] % 2 else "web_search"
        return AIMessage("", tool_calls=[{"name": name, "args": {"query": "q"}, "id": f"c{calls['agent']}"}])
    return RunnableLambda(reply)


async def never_grounded(prompt):
    calls["ground"] += 1
    return GroundCheck(grounded=False, issue="made up")


node.agent_model = always_call_tools
node.answer_llm = streaming_answer("final answer [1]")
chain.chain = RunnableLambda(lambda inputs: calls.__setitem__("rag", calls["rag"] + 1) or "rag answer [1]")
node.ground_llm = RunnableLambda(never_grounded)

from app.main import app  # noqa: E402


def sse_events(text: str) -> list[tuple[str, object]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_agent_limits_and_stream() -> None:
    with TestClient(app) as client:
        res = client.post("/chat/stream", json={"message": "compare docs and web", "mode": "agent"})
        history = client.get(f"/chat/{sse_events(res.text)[-1][1]['thread_id']}/history").json()["messages"]
    events = sse_events(res.text)
    names = [e for e, _ in events]

    # Tool loop is capped, then a forced answer; grounding retry runs at most once.
    assert calls["agent"] == settings.max_tool_calls, calls
    assert calls["web"] == settings.max_tool_calls // 2
    assert calls["ground"] == settings.max_answer_retries, calls

    statuses = [d for e, d in events if e == "status"]
    assert statuses[0] == "Searching documents: q" and "Searching the web: q" in statuses
    assert "Improving answer (not fully supported by sources)" in statuses
    assert "".join(d for e, d in events if e == "token").endswith("final answer [1]")

    done = events[-1]
    assert names[-1] == "done", names
    assert done[1]["response"] == "final answer [1]"
    # Citations numbered across the two document searches.
    assert [c["n"] for c in done[1]["sources"]] == [1, 2]

    # Tool calls stay out of the saved conversation.
    assert history == [
        {"role": "user", "text": "compare docs and web"},
        {"role": "assistant", "text": "final answer [1]"},
    ]


def test_ask_retry_bounded() -> None:
    calls["ground"] = calls["rag"] = 0
    with TestClient(app) as client:
        body = client.post("/chat/ask", json={"message": "what is x?"}).json()
    assert calls["rag"] == 1 + settings.max_answer_retries and calls["ground"] == settings.max_answer_retries
    assert body["sources"][0] == {"n": 1, "source": "a.pdf", "page": 2, "snippet": "fact about what is x?"}


def test_read_document_tool() -> None:
    state = {"scratch": [], "sources_filter": None}
    call = lambda source: asyncio.run(tools.read_document.ainvoke(
        {"type": "tool_call", "name": "read_document", "id": "1", "args": {"source": source, "state": state}}
    ))
    found = call("a.pdf")
    assert "[1] (a.pdf p.1)" in found.content and len(found.artifact) == 2
    assert "No document named" in call("missing.pdf").content

    state["sources_filter"] = ["other.pdf"]
    assert "not among the documents selected" in call("a.pdf").content

    note = asyncio.run(node._documents_note({"sources_filter": None}))
    assert "- a.pdf" in note


def make_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    return out + b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objects) + 1, xref)


def test_pdf_and_ids() -> None:
    pages = load_document(make_pdf("Hello PDF world"), "report.pdf")
    assert len(pages) == 1 and "Hello PDF world" in pages[0].page_content
    assert pages[0].metadata == {"source": "report.pdf", "file_type": "pdf", "page": 1}

    chunk = Document(page_content="x", metadata={"source": "report.pdf", "page": 3, "start_index": 10})
    assert chunk_id(chunk) == f"{doc_prefix('report.pdf')}3#10" and chunk_id(chunk).isascii()

    try:
        load_document(b"not a pdf", "bad.pdf")
    except ValueError:
        pass
    else:
        raise AssertionError("bad PDF should be rejected")


def test_upstream_failures() -> None:
    import app.ai.llm as llm_module

    # Web search down: the model gets a plain message instead of an exception.
    async def broken_tavily(_):
        raise ConnectionError("tavily down")

    working, web.tavily_search = web.tavily_search, SimpleNamespace(ainvoke=broken_tavily)
    try:
        assert "unavailable" in asyncio.run(web.web_search.ainvoke({"query": "q"}))
    finally:
        web.tavily_search = working

    # Main model failing: the fallback model answers.
    def down(_):
        raise ConnectionError("model down")

    primary = RunnableLambda(down)
    real_model, real_setting = llm_module._model, settings.llm_fallback_model
    llm_module._model = lambda model, max_tokens: RunnableLambda(lambda _: f"from {model}")
    settings.llm_fallback_model = "backup-model"
    try:
        assert llm_module._with_fallback(primary, 10).invoke("hi") == "from backup-model"
        settings.llm_fallback_model = None
        assert llm_module._with_fallback(primary, 10) is primary
    finally:
        llm_module._model, settings.llm_fallback_model = real_model, real_setting


def test_database_stats() -> None:
    stats = SimpleNamespace(
        dimension=512, total_vector_count=10,
        namespaces={"development": SimpleNamespace(vector_count=7), "other": SimpleNamespace(vector_count=3)},
    )
    real_index, real_list = tools.get_index, tools.list_documents
    tools.list_documents = fake_list_documents
    try:
        tools.get_index = lambda: SimpleNamespace(describe_index_stats=lambda: stats)
        out = asyncio.run(tools.database_stats.ainvoke({}))
        assert "Total vectors (chunks): 10" in out and "dimension 512" in out, out
        assert "development: 7, other: 3" in out and "- a.pdf: 3 chunks" in out, out
        assert "~0.02 MB" in out, out

        def down():
            raise ConnectionError("pinecone down")
        tools.get_index = lambda: SimpleNamespace(describe_index_stats=down)
        assert "unavailable" in asyncio.run(tools.database_stats.ainvoke({}))
    finally:
        tools.get_index, tools.list_documents = real_index, real_list


if __name__ == "__main__":
    test_agent_limits_and_stream()
    test_ask_retry_bounded()
    test_read_document_tool()
    test_pdf_and_ids()
    test_upstream_failures()
    test_database_stats()
    print("ok")
