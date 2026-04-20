from pydantic_settings import BaseSettings
from pydantic import Field
from app.config.env_loader import load_environment

load_environment()

class Settings(BaseSettings):
    # --- Environment ---
    env: str = Field(default="dev")

    # --- API ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # --- Logging ---
    log_level: str = Field(default="INFO")
    enable_file_logs: bool = Field(default=True)

    # --- Redis ---
    redis_url: str = Field(..., description="Redis connection URL")

    # --- Qdrant ---
    qdrant_url: str = Field(..., description="Qdrant endpoint")
    qdrant_api_key: str | None = Field(default=None)

    # --- OpenAI / LLM ---
    openai_api_key: str = Field(...)

    # --- Metrics ---
    metrics_port: int = Field(default=9000)

    # --- Security ---
    api_key: str | None = Field(default=None)
    rate_limit_per_minute: int = Field(default=60)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
