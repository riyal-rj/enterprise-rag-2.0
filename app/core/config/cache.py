"""Upstash Redis cache configuration and per-artifact TTLs."""

from __future__ import annotations

from pydantic import Field, SecretStr

from app.core.config.base import EnvBaseSettings


class CacheSettings(EnvBaseSettings):
    """Redis connection and TTLs (seconds) for each cached artifact type."""

    upstash_redis_url: str = Field(default="")
    upstash_redis_token: SecretStr = Field(default=SecretStr(""))

    cache_ttl_embeddings: int = Field(default=604_800, ge=0)
    cache_ttl_rag: int = Field(default=3_600, ge=0)
    cache_ttl_sql_gen: int = Field(default=86_400, ge=0)
    cache_ttl_sql_result: int = Field(default=900, ge=0)
    cache_ttl_intent: int = Field(default=86_400, ge=0)
