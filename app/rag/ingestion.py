import asyncio
import hashlib

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from app.ai.embedding import embeddings
from app.config import settings
from app.vector_store.pinecone import get_index

vector_store = PineconeVectorStore(
    index_name=settings.pinecone_index_name,
    embedding=embeddings,
    namespace=settings.pinecone_namespace,
    pinecone_api_key=settings.pinecone_api_key
)

NAMESPACE = settings.pinecone_namespace
ID_PREFIX = "doc-"


def doc_prefix(source: str) -> str:
    # Pinecone ids must be ASCII; hash the filename so any name works and all
    # chunks of one file share a prefix (used to list and delete a document).
    return f"{ID_PREFIX}{hashlib.sha1(source.encode()).hexdigest()[:12]}#"


def chunk_id(chunk: Document) -> str:
    meta = chunk.metadata
    return f"{doc_prefix(meta['source'])}{meta.get('page', 1)}#{meta.get('start_index', 0)}"


def _ids_with_prefix(prefix: str) -> list[str]:
    return [i for page in get_index().list(prefix=prefix, namespace=NAMESPACE) for i in page]


async def delete_document(source: str) -> int:
    ids = await asyncio.to_thread(_ids_with_prefix, doc_prefix(source))
    if ids:
        index = get_index()
        for start in range(0, len(ids), 1000):  # Pinecone delete limit per call
            await asyncio.to_thread(index.delete, ids=ids[start:start + 1000], namespace=NAMESPACE)
    return len(ids)


def _list_documents() -> list[dict]:
    counts: dict[str, list[str]] = {}
    for vector_id in _ids_with_prefix(ID_PREFIX):
        counts.setdefault(vector_id.split("#", 1)[0], []).append(vector_id)
    if not counts:
        return []

    # One metadata fetch per document to recover its filename.
    first_ids = [ids[0] for ids in counts.values()]
    fetched = get_index().fetch(ids=first_ids, namespace=NAMESPACE).vectors
    return sorted(
        (
            {"source": fetched[ids[0]].metadata.get("source", "unknown"), "chunks": len(ids)}
            for ids in counts.values()
            if ids[0] in fetched
        ),
        key=lambda doc: doc["source"].lower(),
    )


async def list_documents() -> list[dict]:
    return await asyncio.to_thread(_list_documents)


def _read_document(source: str, max_chunks: int) -> list[Document]:
    def position(vector_id: str) -> tuple[int, int]:
        _, page, start = vector_id.rsplit("#", 2)
        return int(page), int(start)

    ids = sorted(_ids_with_prefix(doc_prefix(source)), key=position)[:max_chunks]
    if not ids:
        return []
    vectors = get_index().fetch(ids=ids, namespace=NAMESPACE).vectors
    return [
        Document(
            page_content=vectors[i].metadata.get("text", ""),
            metadata={k: v for k, v in vectors[i].metadata.items() if k != "text"},
        )
        for i in ids
        if i in vectors
    ]


async def read_document(source: str, max_chunks: int = 8) -> list[Document]:
    """First chunks of a document in reading order (for summaries/overviews)."""
    return await asyncio.to_thread(_read_document, source, max_chunks)


async def store_chunk(chunks: list[Document]) -> list[str]:
    if not chunks:
        raise ValueError("No document chunk to store")

    # Replace any previous version of the file (a shorter new version would
    # otherwise leave stale trailing chunks behind).
    await delete_document(chunks[0].metadata["source"])

    return await vector_store.aadd_documents(chunks, ids=[chunk_id(c) for c in chunks])
