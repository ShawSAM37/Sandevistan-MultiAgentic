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

    azure_openai_answer_deployment: str | None = None
    azure_openai_guardrail_deployment: str | None = None
    azure_openai_planner_deployment: str | None = None
    azure_openai_critic_deployment: str | None = None
    azure_openai_summarizer_deployment: str | None = None
    azure_openai_fallback_chat_deployment: str | None = None

    azure_openai_api_version: str = "2024-02-01"

    vector_dimensions: int = 1024

    default_top_k: int = 10
    max_context_chars: int = 12000
    max_chars_per_document: int = 2500
    max_revision_count: int = 1
    max_search_plans: int = 3
    max_llm_calls_per_request: int = 6
    request_timeout_seconds: int = 60

    answer_max_completion_tokens: int = 800
    guardrail_max_completion_tokens: int = 200
    critic_max_completion_tokens: int = 1000
    revision_max_completion_tokens: int = 1000

    openai_chat_min_interval_seconds: float = 8.0
    openai_chat_retry_count: int = 2
    openai_chat_retry_base_seconds: float = 5.0
    openai_chat_retry_jitter_seconds: float = 2.0

    conversation_summary_max_chars: int = 2000
    max_recent_turns: int = 4

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






