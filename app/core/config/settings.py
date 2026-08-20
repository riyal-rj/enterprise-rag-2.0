"""Composed application settings and the process-wide accessor.

Each domain settings group (:class:`~app.core.config.llm.LLMSettings`,
:class:`~app.core.config.cache.CacheSettings`, etc.) is an independent
``BaseSettings`` subclass that can be constructed and unit-tested in
isolation. :class:`Settings` composes them into a single object; use
:func:`get_settings` to obtain it so call sites depend on an injectable
accessor rather than reaching for a module-level global directly.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.config.cache import CacheSettings
from app.core.config.cors import CorsSettings
from app.core.config.database import INSECURE_DEFAULT_DSN, DatabaseSettings
from app.core.config.environment import Environment, EnvironmentSettings
from app.core.config.external_apis import ExternalAPISettings
from app.core.config.ingestion import IngestionSettings
from app.core.config.llm import LLMSettings
from app.core.config.log_settings import LoggingSettings
from app.core.config.rag_features import RAGFeatureSettings
from app.core.config.safety import SafetySettings
from app.core.config.security import AuthSettings, RateLimitSettings
from app.core.config.sql_settings import SQLFeatureSettings
from app.core.config.storage import StorageSettings
from app.core.config.token_budget import TokenBudgetSettings
from app.core.config.vector_store import QdrantSettings


class Settings(BaseModel):
    """Root settings object aggregating every configuration domain."""

    model_config = ConfigDict(frozen=True)

    environment: Environment = Field(default_factory=lambda: EnvironmentSettings().environment)

    llm: LLMSettings = Field(default_factory=LLMSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    cors: CorsSettings = Field(default_factory=CorsSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    external_apis: ExternalAPISettings = Field(default_factory=ExternalAPISettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    token_budget: TokenBudgetSettings = Field(default_factory=TokenBudgetSettings)
    safety: SafetySettings = Field(default_factory=SafetySettings)
    rag: RAGFeatureSettings = Field(default_factory=RAGFeatureSettings)
    sql: SQLFeatureSettings = Field(default_factory=SQLFeatureSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @model_validator(mode="after")
    def _enforce_production_hardening(self) -> "Settings":
        """Fail fast on insecure or missing config when running in prod."""
        if self.environment is not Environment.PRODUCTION:
            return self

        missing = []
        if not self.llm.openai_api_key.get_secret_value():
            missing.append("OPENAI_API_KEY")
        if not self.auth.jwt_secret.get_secret_value():
            missing.append("JWT_SECRET")
        if self.database.database_url == INSECURE_DEFAULT_DSN:
            missing.append("DATABASE_URL (still using the local dev default)")

        if missing:
            raise ValueError(
                "Refusing to start in production with insecure/missing "
                f"configuration: {', '.join(missing)}"
            )

        # Guardrails: "monitor" mode observes findings but never applies a
        # BLOCK/REDACT decision (see app.guardrails.policy.GuardrailPolicy.decide)
        # - an acceptable posture for evaluating false-positive rates in
        # non-production, but never while a route that can execute SQL or
        # fetch external web content is enabled, since those routes have no
        # enforcement backstop of their own if guardrails are only watching.
        if self.safety.guardrail_mode_default == "monitor" and (
            self.sql.sql_enabled_by_default or self.rag.crag_web_enabled_by_default
        ):
            raise ValueError(
                "Refusing to start in production with guardrail mode 'monitor' "
                "while Text-to-SQL or CRAG web correction is enabled by default - "
                "monitor mode observes but does not block, which is not an "
                "acceptable posture for either feature at rollout time."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide, cached :class:`Settings` instance.

    Cached so settings are parsed and validated once per process. Tests can
    override configuration by patching environment variables and calling
    ``get_settings.cache_clear()`` before the next call, or by using
    dependency-injection overrides (e.g. FastAPI's
    ``app.dependency_overrides``) instead of calling this function directly.
    """
    return Settings()
