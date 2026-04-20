from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App / service
    app_name: str = "multi-agent-ecommerce-analyst"
    app_version: str = "1.0.0"
    environment: str = "development"

    # API / UI ports
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_port: int = 8501
    metrics_port: int = 8001

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Qdrant
    qdrant_mode: str = "local"  # local or server
    qdrant_storage_path: str = "artifacts/qdrant_storage"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "review_embeddings"

    # Observability
    otel_service_name: str = "multi-agent-ecommerce-analyst"
    otel_traces_exporter_mode: str = "console"  # console or otlp
    otel_exporter_otlp_endpoint: str = "http://localhost:4318"

    # Models
    llm_model: str = "gpt-4.1-mini"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    aspect_sentiment_backend: str = "zero_shot"

    # Cache
    analysis_cache_ttl_seconds: int = 3600
    retrieval_cache_ttl_seconds: int = 1800

    # Security / future auth
    api_key: str | None = None

    # Optional external provider keys
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")


settings = Settings()