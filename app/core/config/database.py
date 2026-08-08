"""Relational database configuration."""

from __future__ import annotations

from pydantic import Field, field_validator

from app.core.config.base import EnvBaseSettings

INSECURE_DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/adv_rag"


class DatabaseSettings(EnvBaseSettings):
    """Postgres connection string used by the core RAG and Vanna pipelines."""

    database_url: str = Field(default=INSECURE_DEFAULT_DSN)

    @field_validator("database_url")
    @classmethod
    def _must_be_postgres_dsn(cls, value: str) -> str:
        if not value.startswith(("postgresql://", "postgresql+psycopg://")):
            raise ValueError("database_url must be a postgresql:// DSN")
        return value
