from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "azure"
    app_name: str = "rotary-multi-agent-rag"
    log_level: str = "INFO"

    azure_search_endpoint: str | None = None
    azure_search_admin_key: str | None = None
    azure_search_index_name: str = "rotary-instruction-manuals"
    azure_search_semantic_config: str = "rotary-instruction-manuals-semantic-config"
    azure_search_api_version: str = "2024-07-01"

    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_embedding_deployment: str | None = None
    azure_openai_chat_deployment: str | None = None
    azure_openai_api_version: str = "2024-02-01"

    vector_dimensions: int = 1024

    default_top_k: int = 10
    max_context_chars: int = 50000
    max_chars_per_document: int = 10000
    max_revision_count: int = 1
    max_search_plans: int = 3
    max_llm_calls_per_request: int = 6
    request_timeout_seconds: int = 60

    enable_title_vector_search: bool = True
    enable_semantic_ranker: bool = True

    azure_storage_account: str | None = None
    eval_results_container: str = "eval-results"

    backend_base_url: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

