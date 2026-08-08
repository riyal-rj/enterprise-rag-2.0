"""Context-window token budgeting configuration."""

from __future__ import annotations

from pydantic import Field

from app.core.config.base import EnvBaseSettings


class TokenBudgetSettings(EnvBaseSettings):
    """Reserves that bound prompt construction within the model context window."""

    max_input_tokens: int = Field(default=3_000, gt=0)
    reserved_context_tokens: int = Field(default=1_000, ge=0)
    reserved_output_tokens: int = Field(default=1_000, ge=0)
