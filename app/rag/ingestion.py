import hashlib

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from app.ai.embedding import embeddings
from app.config import settings

vector_store = PineconeVectorStore(
    index_name=settings.pinecone_index_name,
    embedding=embeddings,
    namespace=settings.pinecone_namespace,
    pinecone_api_key=settings.pinecone_api_key
)

def chunk_id(chunk: Document) -> str:
    # Deterministic id: re-uploading the same file overwrites instead of duplicating.
    key = f"{chunk.metadata.get('source')}:{chunk.metadata.get('start_index')}"
    return hashlib.sha256(key.encode()).hexdigest()

async def store_chunk(chunks: list[Document]) -> list[str]:
    if not chunks:
        raise ValueError("No document chunk to store")
    
    return await vector_store.aadd_documents(chunks, ids=[chunk_id(c) for c in chunks])
