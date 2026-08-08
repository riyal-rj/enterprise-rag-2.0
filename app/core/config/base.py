"""Shared base class for every settings group in :mod:`app.core.config`."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvBaseSettings(BaseSettings):
    """Common ``.env``/environment loading behavior for all settings groups.

    Field names are matched against environment variables case-insensitively
    (e.g. ``openai_api_key`` <-> ``OPENAI_API_KEY``), so existing deployment
    configs keep working unchanged when settings are split into groups.
    Instances are frozen to prevent accidental mutation once loaded.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )
