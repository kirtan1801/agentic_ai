from pinecone import Pinecone
from app.config import settings

pinecone_client = Pinecone(
    api_key=settings.pinecone_api_key
)

def get_index():
    return pinecone_client.Index(settings.pinecone_index_name) 