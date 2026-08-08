"""Guardrail thresholds for prompt/output safety validation."""

from __future__ import annotations

from pydantic import Field

from app.core.config.base import EnvBaseSettings


class SafetySettings(EnvBaseSettings):
    """Score thresholds (0-1) used by input/output validation guardrails."""

    prompt_injection_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    toxicity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    output_toxicity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_validation_retries: int = Field(default=2, ge=0)
