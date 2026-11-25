"""Settings originate from container environment; no secrets are embedded in source."""

from functools import lru_cache
from typing import Self
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL

INFERENCE_READINESS_TIMEOUT_SECONDS = 8.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    db_host: str = "postgres"
    db_port: int = Field(default=5432, ge=1, le=65535)
    db_name: str = "ddu_game"
    db_user: str = "ddu_app"
    db_password: str = ""
    inference_gateway_base_url: str = Field(
        default="http://ddu-gateway:8080", min_length=1, max_length=2048
    )
    inference_gateway_token: str = Field(default="", max_length=512, repr=False)
    llm_model_alias: str = Field(default="ddu-narrator", min_length=1, max_length=128)
    llm_enable_thinking: bool | None = None
    llm_completion_timeout_seconds: float = Field(default=240.0, ge=1.0, le=600.0)
    luna_model_id: str = Field(default="gpt-6-luna", min_length=1, max_length=128)
    rag_embed_base_url: str = Field(default="http://ddu-embed:8080", min_length=1, max_length=2048)
    rag_embed_model_id: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        min_length=1,
        max_length=160,
    )
    rag_embed_timeout_seconds: float = Field(default=15.0, ge=0.1, le=120.0)
    rag_rerank_enabled: bool = False
    rag_rerank_base_url: str | None = Field(default=None, max_length=2048)
    rag_rerank_model_id: str = Field(
        default=(
            "Alibaba-NLP/gte-multilingual-reranker-base@8215cf04918ba6f7b6a62bb44238ce2953d8831c"
            "+code@40ced75c3017eb27626c9d4ea981bde21a2662f4"
        ),
        min_length=1,
        max_length=160,
    )
    rag_rerank_timeout_seconds: float = Field(default=20.0, ge=0.1, le=120.0)
    rag_candidate_limit: int = Field(default=20, ge=2, le=100)
    rag_rrf_k: int = Field(default=60, ge=1, le=1000)
    rag_min_lexical_score: float = Field(default=0.03, ge=0.0, le=1.0)
    rag_min_dense_similarity: float = Field(default=0.60, ge=-1.0, le=1.0)
    rag_min_rerank_score: float = Field(default=0.20, ge=0.0, le=1.0)
    worker_poll_seconds: float = Field(default=10.0, ge=1.0, le=600.0)
    worker_stale_seconds: int = Field(default=45, ge=2, le=3600)
    turn_lease_seconds: int = Field(default=60, ge=1, le=600)
    turn_lease_renew_interval_seconds: float = Field(default=15.0, gt=0.0, le=300.0)
    turn_context_timeout_seconds: float = Field(default=30.0, ge=0.1, le=300.0)
    turn_inference_timeout_seconds: float = Field(default=220.0, ge=1.0, le=600.0)
    turn_total_timeout_seconds: float = Field(default=1380.0, ge=2.0, le=1800.0)

    @field_validator("inference_gateway_base_url")
    @classmethod
    def validate_inference_gateway_base_url(cls, value: str) -> str:
        value = value.strip()
        try:
            parsed = urlsplit(value)
            parsed.port  # Accessing port validates its syntax and range.
        except ValueError as error:
            raise ValueError("inference_gateway_base_url must be a valid HTTP(S) URL") from error
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("inference_gateway_base_url must use HTTP or HTTPS and include a host")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("inference_gateway_base_url must not include credentials")
        return value.rstrip("/")

    @field_validator("inference_gateway_token")
    @classmethod
    def validate_inference_gateway_token(cls, value: str) -> str:
        value = value.strip()
        if value and len(value) < 24:
            raise ValueError("inference_gateway_token must be at least 24 characters when configured")
        return value

    @field_validator("llm_model_alias")
    @classmethod
    def validate_llm_model_alias(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("llm_model_alias must not be empty")
        return value

    @field_validator("luna_model_id")
    @classmethod
    def validate_luna_model_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("luna_model_id must not be empty")
        return value

    @field_validator("rag_embed_base_url")
    @classmethod
    def validate_rag_embed_base_url(cls, value: str) -> str:
        return cls._validated_service_url(value, "rag_embed_base_url")

    @field_validator("rag_rerank_base_url", mode="before")
    @classmethod
    def empty_rerank_url_is_unset(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("rag_rerank_base_url")
    @classmethod
    def validate_rag_rerank_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return cls._validated_service_url(value, "rag_rerank_base_url")

    @field_validator("rag_embed_model_id", "rag_rerank_model_id")
    @classmethod
    def validate_rag_model_ids(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("RAG model ID must not be empty")
        return value

    @staticmethod
    def _validated_service_url(value: str, field_name: str) -> str:
        value = value.strip()
        try:
            parsed = urlsplit(value)
            parsed.port
        except ValueError as error:
            raise ValueError(f"{field_name} must be a valid HTTP(S) URL") from error
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"{field_name} must use HTTP or HTTPS and include a host")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError(f"{field_name} must not include credentials")
        return value.rstrip("/")

    @field_validator("llm_enable_thinking", mode="before")
    @classmethod
    def empty_thinking_setting_is_unset(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def validate_worker_timing(self) -> Self:
        max_probe_seconds = 2 * INFERENCE_READINESS_TIMEOUT_SECONDS
        if self.worker_stale_seconds <= self.worker_poll_seconds + max_probe_seconds:
            raise ValueError(
                "worker_stale_seconds must exceed worker_poll_seconds plus the "
                "health and model-list readiness timeouts; allow extra time for database writes"
            )
        return self

    @model_validator(mode="after")
    def validate_turn_timing(self) -> Self:
        if self.turn_lease_renew_interval_seconds * 2 >= self.turn_lease_seconds:
            raise ValueError("turn lease renewal interval must be less than half the lease")
        # Cover the six-call ceiling: bounded tool planning, repair, final corrections,
        # and proposal generation. Tool queries themselves run concurrently.
        minimum_total = (
            self.turn_context_timeout_seconds + 6 * self.turn_inference_timeout_seconds + 5
        )
        if self.turn_total_timeout_seconds < minimum_total:
            raise ValueError(
                "turn total timeout must allow context and six bounded model calls"
            )
        return self

    @model_validator(mode="after")
    def validate_rag_services(self) -> Self:
        if self.rag_rerank_enabled and self.rag_rerank_base_url is None:
            raise ValueError("rag_rerank_base_url is required when rag_rerank_enabled is true")
        return self

    @property
    def database_url(self) -> URL:
        # URL.create safely quotes punctuation in passwords, unlike manual DSN concatenation.
        return URL.create(
            "postgresql+asyncpg",
            username=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
