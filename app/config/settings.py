from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.config.env_loader import load_environment

load_environment()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment
    env: str = "dev"
    environment: str = "development"

    # App / service
    app_name: str = "multi-agent-ecommerce-analyst"
    app_version: str = "1.0.0"

    # API / UI ports
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_port: int = 8501
    metrics_port: int = 8001

    # Logging
    log_level: str = "INFO"
    enable_file_logs: bool = True

    # Redis
    redis_url: str = Field(..., description="Redis connection URL")

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_mode: str = "local"
    qdrant_storage_path: str = "artifacts/qdrant_storage"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "review_embeddings"

    # Observability
    otel_service_name: str = "multi-agent-ecommerce-analyst"
    otel_traces_exporter_mode: str = "console"
    otel_exporter_otlp_endpoint: str = "http://localhost:4318"
    tracing_sample_rate: float = 1.0

    # Models / LLM
    llm_model: str = "gpt-4.1-mini"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    aspect_sentiment_backend: str = "zero_shot"
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    # Cache
    analysis_cache_ttl_seconds: int = 3600
    retrieval_cache_ttl_seconds: int = 1800

    # Security
    api_key: str | None = None
    rate_limit_per_minute: int = 60

    rate_limit_overrides: dict[str, int] = Field(default_factory=dict)

settings = Settings()