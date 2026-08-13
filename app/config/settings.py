from __future__ import annotations

import json

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
    # New, additive field -- separate from api_key above, not a
    # replacement for it. Comma-separated, same format and parsing
    # convention as api_key, so no new pattern to learn. A key listed
    # here is automatically valid everywhere an api_key would be too
    # (see auth_middleware.py's is_admin/is_analyst logic) -- it doesn't
    # need to also appear in api_key to be let in at all.
    admin_api_keys: str | None = None
    rate_limit_per_minute: int = 60
    cors_allowed_origins: str = ""
    # Same comma-separated-string pattern as cors_allowed_origins above --
    # DNS-rebinding protection for the MCP endpoint specifically, needed
    # because omitting/misconfiguring this produces a 421 rejection on
    # every single tool call, confirmed directly by testing. Real
    # deployment hostname isn't known at build time, hence a setting
    # rather than a hardcoded value.
    mcp_allowed_hosts: str = ""
    # All empty/None by default -- deliberately opt-in. No real external
    # price-check MCP server has actually been chosen yet; these are
    # placeholders buy_decision_agent checks at construction time, and
    # it behaves exactly as it did before this feature existed until
    # someone actually sets price_check_mcp_server_url to something real.
    price_check_transport: str = "http"
    price_check_mcp_server_url: str | None = None
    # stdio-specific: needed because research into real, existing price-
    # check MCP servers (Keepa-backed ones specifically) found every
    # implementation runs as a locally-spawned subprocess, not a remote
    # HTTP endpoint -- command/args match the same shape those servers'
    # own install instructions use (e.g. command="npx",
    # args=["keepa-mcp-server@latest"]).
    price_check_command: str | None = None
    price_check_args_raw: str = ""
    # JSON string, not comma-separated -- this is key-value config (e.g.
    # an API key env var), not a flat list like args/hosts above.
    price_check_env_json: str = ""
    price_check_tool_name: str = "check_price"
    price_check_query_arg_name: str = "query"
    price_check_price_field_name: str = "current_price"

    @property
    def price_check_args(self) -> list[str]:
        """Comma-separated price-check subprocess args from env/config, parsed into a list."""
        return [a.strip() for a in self.price_check_args_raw.split(",") if a.strip()]

    @property
    def price_check_env(self) -> dict[str, str] | None:
        """JSON-encoded price-check subprocess env vars (e.g. an API key), parsed into a dict."""
        if not self.price_check_env_json:
            return None
        try:
            return json.loads(self.price_check_env_json)
        except json.JSONDecodeError:
            return None

    rate_limit_overrides: dict[str, int] = Field(default_factory=dict)

    @property
    def cors_allowed_origins_list(self) -> list[str]:
        """Comma-separated origins from env/config, parsed into a list.

        Defaults to empty (no origins allowed) rather than "*" -- CORS should
        fail closed, since an empty allowlist is safe by default and just
        needs to be set explicitly per environment.
        """
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    @property
    def mcp_allowed_hosts_list(self) -> list[str]:
        """Comma-separated MCP allowed hosts from env/config, parsed into a list."""
        return [h.strip() for h in self.mcp_allowed_hosts.split(",") if h.strip()]

settings = Settings()