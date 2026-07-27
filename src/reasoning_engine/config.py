from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    app_name: str = "Open Reasoning Engine"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_request_timeout: float = 180.0

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 512

    vector_store: Literal["qdrant", "milvus"] = "qdrant"
    collection_name: str = "reasoning_engine"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str | None = None

    similarity_top_k: int = Field(default=5, ge=1, le=20)
    chunk_size: int = Field(default=700, ge=128, le=4096)
    chunk_overlap: int = Field(default=100, ge=0, le=1024)
    max_retries: int = Field(default=1, ge=0, le=3)
    min_relevance_score: float = Field(default=0.25, ge=0, le=1)

    @field_validator("qdrant_api_key", "milvus_token", mode="before")
    @classmethod
    def empty_string_is_none(cls, value: object) -> object:
        return None if value == "" else value


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()
