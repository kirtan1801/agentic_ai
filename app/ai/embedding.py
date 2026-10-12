from langchain_openai import OpenAIEmbeddings

from app.config import settings

embeddings =  OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=settings.llm_api_key,
    dimensions=512,
)