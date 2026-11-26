"""Centralized application settings loaded from environment variables and .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed and validated application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = "expense-ai"
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    # Provider & Model Configuration
    llm_provider: Literal["openai", "claude", "bedrock", "gemini"] = Field(
        default="claude",
        validation_alias="LLM_PROVIDER",
    )
    # temperature: float = Field(default=0.0, validation_alias="TEMPERATURE")
    max_retries: int = Field(default=3, validation_alias="MAX_RETRIES")
    max_tokens: int = Field(default=4096, validation_alias="MAX_TOKENS")

    # OpenAI
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model_id: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL_ID")

    # Anthropic / Claude
    claude_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
    )
    claude_model_id: str = Field(default="claude-sonnet-4-6", validation_alias="CLAUDE_MODEL_ID")

    # AWS / Bedrock / S3
    aws_region: str = Field(default="us-east-1", validation_alias="AWS_REGION")
    aws_access_key_id: str | None = Field(default=None, validation_alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: SecretStr | None = Field(
        default=None, validation_alias="AWS_SECRET_ACCESS_KEY"
    )
    aws_session_token: SecretStr | None = Field(
        default=None, validation_alias="AWS_SESSION_TOKEN"
    )
    bedrock_model_id: str = Field(
        default="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        validation_alias="BEDROCK_MODEL_ID",
    )
    bedrock_embedding_model_id: str = Field(
        default="amazon.titan-embed-text-v2:0",
        validation_alias="BEDROCK_EMBEDDING_MODEL_ID",
    )

    # Google / Gemini
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )
    gemini_model_id: str = Field(
        default="gemini-4-neon",
        validation_alias="GEMINI_MODEL_ID",
    )
    gemini_thinking_level: Literal["MINIMAL", "LOW", "MEDIUM", "HIGH", "XHIGH"] | None = Field(
        default=None,
        validation_alias="GEMINI_THINKING_LEVEL",
    )

    # TypeSafe AI
    typesafe_api_key: SecretStr | None = Field(default=None, validation_alias="TYPESAFE_API_KEY")
    typesafe_model: str = Field(default="jev-latest", validation_alias="TYPESAFE_DEFAULT_MODEL")
    typesafe_max_concurrency: int = Field(
        default=5, validation_alias="TYPESAFE_MAX_CONCURRENCY"
    )

    # Storage Configuration
    storage_backend: Literal["local", "s3"] | None = Field(
        default=None, validation_alias="STORAGE_BACKEND"
    )
    local_mode: bool | None = Field(default=None, validation_alias="LOCAL_MODE")
    use_local: bool | None = Field(default=None, validation_alias="USE_LOCAL")
    bucket_name: str | None = Field(default=None, validation_alias="BUCKET_NAME")

    # Local Directories
    base_dir: Path = Field(default_factory=Path.cwd)
    pdf_dir: str = Field(default="pdf", validation_alias="PDF_DIR")
    transactions_dir: str = Field(default="transactions", validation_alias="TRANSACTIONS_DIR")
    year_total_dir: str = Field(default="year_total", validation_alias="YEAR_TOTAL_DIR")

    # Execution Mode
    single_file_mode: bool = Field(default=False, validation_alias="SINGLE_FILE_MODE")
    single_file_mode_pdf_name: str = Field(
        default="20251125-statements-8506-.pdf",
        validation_alias="SINGLE_FILE_MODE_PDF_NAME",
    )

    @field_validator("gemini_thinking_level", mode="before")
    @classmethod
    def _normalize_gemini_thinking_level(cls, v: object) -> object:
        if isinstance(v, str):
            stripped = v.strip()
            return stripped.upper() if stripped else None
        return v

    @model_validator(mode="after")
    def _resolve_storage_backend(self) -> "Settings":
        if self.storage_backend is None:
            if self.local_mode is True or self.use_local is True:
                self.storage_backend = "local"
            elif self.local_mode is False and self.use_local is False and self.bucket_name:
                self.storage_backend = "s3"
            else:
                self.storage_backend = "local"
        return self

    @property
    def active_model_id(self) -> str:
        """Return the model ID corresponding to the active LLM provider."""
        match self.llm_provider:
            case "openai":
                return self.openai_model_id
            case "claude":
                return self.claude_model_id
            case "bedrock":
                return self.bedrock_model_id
            case "gemini":
                return self.gemini_model_id


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
