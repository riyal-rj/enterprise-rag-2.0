"""Request/response schemas for the ``/admin/security-events`` route."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class SecurityEventResponse(BaseModel):
    id: int
    actor: str | None
    action: str
    stage: str
    category: str | None
    mode: str
    changes: dict[str, Any]
    reason: str | None
    created_at: datetime


class SecurityEventsResponse(BaseModel):
    """Response for ``GET /admin/security-events``."""

    items: list[SecurityEventResponse]
