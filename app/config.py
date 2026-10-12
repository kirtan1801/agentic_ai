from pydantic_settings import BaseSettings, SettingsConfigDict


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

    max_upload_bytes: int = 2 * 1024 * 1024
    rate_limit_per_minute: int = 20

    max_input_chars: int = 2000
    guard_moderation: bool = True
    guard_nemo: bool = False
    guard_llm_judge: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
