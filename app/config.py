from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# pydantic-settings reads .env but doesn't export it; LangChain/LangSmith read
# LANGSMITH_* from os.environ, so load .env into the environment first.
load_dotenv()


class Settings(BaseSettings):
    llm_provider: str
    llm_model: str
    llm_api_key: str
    # Cheap model for routing decisions and guardrail checks; defaults to llm_model.
    llm_router_model: str | None = None
    llm_max_tokens: int = 3000

    tavily_api_key: str

    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_namespace: str = "development"
    retrieval_min_score: float = 0.35
    rerank_enabled: bool = True
    rerank_model: str = "bge-reranker-v2-m3"
    rerank_min_score: float = 0.05

    # Agent limits: tool calls per turn, and regenerations of an ungrounded answer.
    max_tool_calls: int = 4
    max_answer_retries: int = 1

    max_upload_bytes: int = 5 * 1024 * 1024
    rate_limit_per_minute: int = 20

    max_input_chars: int = 2000
    guard_moderation: bool = True
    guard_nemo: bool = False
    guard_llm_judge: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # .env also holds keys read by other libraries (e.g. LANGSMITH_*).
        extra="ignore",
    )


settings = Settings()
