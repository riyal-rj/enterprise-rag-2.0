"""Credentials for auxiliary third-party APIs (e.g. web search)."""

from __future__ import annotations

from pydantic import Field, SecretStr

from app.core.config.base import EnvBaseSettings


class ExternalAPISettings(EnvBaseSettings):
    """Credentials for external services not tied to a specific pipeline stage."""

    tavily_api_key: SecretStr = Field(default=SecretStr(""))

    # Comma-separated, not list[str] - same reasoning as
    # CorsSettings.cors_allowed_origins: pydantic-settings parses list-typed
    # env vars as JSON, which is an awkward footgun for
    # CRAG_ALLOWED_REGULATORY_DOMAINS=rbi.org.in,sebi.gov.in,npci.org.in.
    # Empty by default - CRAG web correction stays unavailable (see
    # app.api.deps.get_regulatory_web_retriever) until Legal/Compliance
    # approves a real allowlist for this deployment.
    crag_allowed_regulatory_domains: str = Field(default="")
    crag_web_max_results: int = Field(default=5, ge=1, le=10)

    @property
    def allowed_regulatory_domains(self) -> frozenset[str]:
        return frozenset(
            domain.strip().casefold()
            for domain in self.crag_allowed_regulatory_domains.split(",")
            if domain.strip()
        )
