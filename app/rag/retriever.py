from langchain_core.documents import Document

from app.config import settings
from app.rag.ingestion import vector_store

async def retrieve_docs(query: str, k: int = 4) -> list[Document]:
    query = query.strip()

    if not query:
        raise ValueError("Query can not be empty")
    
    if k < 1:
        raise ValueError("k must be greater then zero.")
    
    result = await vector_store.asimilarity_search_with_score(query, k=k)

    # Drop weak matches: fewer context tokens, and no LLM call when nothing is relevant.
    return [
        document
        for document, score in result
        if score >= settings.retrieval_min_score
    ]


def format_context(documents: list[Document]) -> str:
    # Retrieved text is untrusted: fence it so the model treats it as data.
    return "\n\n".join(
        f"<untrusted>\n{doc.page_content}\n</untrusted>" for doc in documents
    )
