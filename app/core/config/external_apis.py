"""Credentials for auxiliary third-party APIs (e.g. web search)."""

from __future__ import annotations

from pydantic import Field, SecretStr

from app.core.config.base import EnvBaseSettings


class ExternalAPISettings(EnvBaseSettings):
    """Credentials for external services not tied to a specific pipeline stage."""

    tavily_api_key: SecretStr = Field(default=SecretStr(""))
