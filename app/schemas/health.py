"""Response schema for ``GET /admin/health``."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class HealthCheckResponse(BaseModel):
    status: Literal["ok", "degraded"]
    qdrant: bool
    postgres: bool
    redis: bool
    openai: bool
    tavily: bool
