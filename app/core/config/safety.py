"""Guardrail thresholds and scanner configuration for the guardrails layer
(``app.guardrails``)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.core.config.base import EnvBaseSettings

GuardrailModeDefault = Literal["enforce", "monitor"]


class SafetySettings(EnvBaseSettings):
    """Score thresholds (0-1) and scanner config used by
    ``app.guardrails``'s input/context/output pipelines."""

    prompt_injection_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    toxicity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    output_toxicity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_validation_retries: int = Field(default=2, ge=0)

    # Seeds rag_ops_config.guardrail_mode's initial row value (see
    # app/seed/migrations/013_add_guardrails_config.sql) and backs the
    # production-hardening check in app.core.config.settings - not read at
    # request time once that DB row exists (same relationship
    # RAGFeatureSettings.crag_enabled_by_default has to rag_ops_config).
    guardrail_mode_default: GuardrailModeDefault = Field(default="enforce")

    # Comma-separated, not list[str] - same reasoning as
    # ExternalAPISettings.crag_allowed_regulatory_domains.
    banned_topics_raw: str = Field(default="")

    # Static, deploy-time-only capability gate - exact analog of
    # ExternalAPISettings.crag_web_guardrails_ready. Defaults False so no
    # deployment silently starts downloading/loading transformer models
    # (PromptInjection, Toxicity, ...) the first time a guardrail pipeline
    # is constructed; flip explicitly once those models are confirmed
    # loadable in this deployment's environment (offline/restricted
    # containers may lack network access to the Hub at runtime). See
    # app.api.deps.get_llm_guard_input_scanner / app.main's startup check.
    scanner_models_ready: bool = Field(default=False)
    llm_guard_scan_timeout_seconds: float = Field(default=5.0, gt=0.0, le=30.0)

    @property
    def banned_topics(self) -> frozenset[str]:
        return frozenset(
            topic.strip() for topic in self.banned_topics_raw.split(",") if topic.strip()
        )
