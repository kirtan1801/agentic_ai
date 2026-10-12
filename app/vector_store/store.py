from langchain_pinecone import PineconeVectorStore

from app.ai.embedding import embeddings
from app.vector_store.pinecone import get_index

def get_vector_store() -> PineconeVectorStore:
    return PineconeVectorStore(
        index=get_index(),
        embedding=embeddings
    )
