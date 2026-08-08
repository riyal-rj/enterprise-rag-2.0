"""Vanna text-to-SQL model configuration."""

from __future__ import annotations

from pydantic import Field

from app.core.config.base import EnvBaseSettings


class VannaSettings(EnvBaseSettings):
    """Model configuration for the Vanna-based SQL generation agent."""

    vanna_model: str = Field(default="gpt-4o")
    vanna_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    vanna_seed: int = Field(default=42)
