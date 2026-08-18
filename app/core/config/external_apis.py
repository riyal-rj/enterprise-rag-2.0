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

    # Static capability gate, independent of crag_web_enabled (the
    # admin-mutable RAG Ops toggle) - see
    # app.api.deps.get_regulatory_web_retriever. Keeps live web correction
    # impossible to switch on in any deployment until the HTTPS/domain/
    # content/timeout guardrails below are actually wired in, regardless of
    # what an admin sets at runtime. Defaults false; flip only once those
    # guardrails are implemented and reviewed for this deployment.
    crag_web_guardrails_ready: bool = Field(default=False)
    crag_web_max_content_chars: int = Field(default=6_000, ge=500, le=20_000)
    crag_web_total_content_chars: int = Field(default=20_000, ge=1_000, le=50_000)
    crag_web_connect_timeout_seconds: float = Field(default=3.0, gt=0.0, le=10.0)
    crag_web_read_timeout_seconds: float = Field(default=8.0, gt=0.0, le=20.0)

    @property
    def allowed_regulatory_domains(self) -> frozenset[str]:
        return frozenset(
            domain.strip().casefold()
            for domain in self.crag_allowed_regulatory_domains.split(",")
            if domain.strip()
        )
