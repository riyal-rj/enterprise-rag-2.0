"""Cross-origin (CORS) configuration for browser-based clients (e.g. the Next.js frontend)."""

from __future__ import annotations

from pydantic import Field

from app.core.config.base import EnvBaseSettings


class CorsSettings(EnvBaseSettings):
    """Origins allowed to call this API from a browser.

    Stored as a comma-separated string (``cors_allowed_origins``) rather
    than a ``list[str]`` field: pydantic-settings parses list-typed env
    vars as JSON, which is an awkward footgun for a simple env var like
    ``CORS_ALLOWED_ORIGINS=http://localhost:3000,https://app.example.com``.
    Defaults to the local Next.js dev server.
    """

    cors_allowed_origins: str = Field(default="http://localhost:3000")

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]
