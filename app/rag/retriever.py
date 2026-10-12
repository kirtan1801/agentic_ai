import asyncio
import logging

from langchain_core.documents import Document

from app.config import settings
from app.rag.ingestion import vector_store
from app.vector_store.pinecone import pinecone_client

logger = logging.getLogger(__name__)

CANDIDATES = 12  # fetched from the vector index, then reranked down to k


def _rerank(query: str, documents: list[Document], k: int) -> list[Document]:
    result = pinecone_client.inference.rerank(
        model=settings.rerank_model,
        query=query,
        documents=[doc.page_content for doc in documents],
        top_n=k,
        return_documents=False,
    )
    return [
        documents[row.index]
        for row in result.data
        if row.score >= settings.rerank_min_score
    ]


async def retrieve_docs(
    query: str,
    k: int = 4,
    sources: list[str] | None = None,
) -> list[Document]:
    query = query.strip()

    if not query:
        raise ValueError("Query can not be empty")
    
    if k < 1:
        raise ValueError("k must be greater then zero.")

    search_filter = {"source": {"$in": sources}} if sources else None
    result = await vector_store.asimilarity_search_with_score(
        query, k=CANDIDATES if settings.rerank_enabled else k, filter=search_filter
    )

    if settings.rerank_enabled and result:
        try:
            # Rerank score doubles as the relevance grade: weak chunks are dropped,
            # so callers can fall back (e.g. to web search) without an extra LLM call.
            return await asyncio.to_thread(_rerank, query, [doc for doc, _ in result], k)
        except Exception:
            logger.warning("Rerank failed; falling back to vector scores", exc_info=True)

    # Drop weak matches: fewer context tokens, and no LLM call when nothing is relevant.
    return [
        document
        for document, score in result[:k]
        if score >= settings.retrieval_min_score
    ]


def format_context(documents: list[Document], start: int = 1) -> str:
    # Numbered for [n] citations. Retrieved text is untrusted: fence it so the
    # model treats it as data.
    return "\n\n".join(
        f"[{n}] ({doc.metadata.get('source', 'unknown')} p.{doc.metadata.get('page', 1)})\n"
        f"<untrusted>\n{doc.page_content}\n</untrusted>"
        for n, doc in enumerate(documents, start=start)
    )


def citations(documents: list[Document], start: int = 1) -> list[dict]:
    return [
        {
            "n": n,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", 1),
            "snippet": doc.page_content[:200],
        }
        for n, doc in enumerate(documents, start=start)
    ]
