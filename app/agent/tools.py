from typing import Annotated
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, InjectedStore
from langgraph.store.base import BaseStore

from app.rag.ingestion import read_document as read_document_chunks
from app.rag.retriever import citations, format_context, retrieve_docs
from app.tools.web_search import web_search

DOCUMENT_TOOLS = {"search_documents", "read_document"}
READ_MAX_CHUNKS = 8  # ~8k characters per read, keeps prompt cost bounded


def _cited(documents: list[Document], state: dict) -> tuple[str, list[dict]]:
    # Continue citation numbering across several document tool calls in one turn.
    start = 1 + sum(
        len(m.artifact or [])
        for m in state["scratch"]
        if isinstance(m, ToolMessage) and m.name in DOCUMENT_TOOLS
    )
    return format_context(documents, start), citations(documents, start)


@tool(response_format="content_and_artifact")
async def search_documents(query: str, state: Annotated[dict, InjectedState]) -> tuple[str, list[dict]]:
    """Search the user's uploaded documents for specific facts. Use a specific, standalone search query."""
    documents = await retrieve_docs(query, sources=state.get("sources_filter"))

    if not documents:
        return "No relevant documents found. Try read_document for overviews, or web_search if the question needs it.", []

    return _cited(documents, state)


@tool(response_format="content_and_artifact")
async def read_document(source: str, state: Annotated[dict, InjectedState]) -> tuple[str, list[dict]]:
    """Read the beginning of one uploaded document, in order. Use for summaries or 'what is this document about'. `source` is the exact filename."""
    allowed = state.get("sources_filter")
    if allowed and source not in allowed:
        return f"'{source}' is not among the documents selected for this question.", []

    documents = await read_document_chunks(source, READ_MAX_CHUNKS)

    if not documents:
        return f"No document named '{source}'. Use one of the filenames listed in your instructions.", []

    return _cited(documents, state)


@tool
async def remember(
    fact: str, state: Annotated[dict, InjectedState], store: Annotated[BaseStore, InjectedStore()]
) -> str:
    """Save one short, lasting fact about the user (name, preferences, goals) so it is known in future chats."""
    if not state.get("user_id"):
        return "Can't save: this chat has no user id."
    # ponytail: no dedupe or editing of facts; add when the 20-fact prompt cap starts to bite.
    await store.aput(("memories", state["user_id"]), uuid4().hex, {"fact": fact})
    return "Saved."


TOOLS = [search_documents, read_document, web_search, remember]
