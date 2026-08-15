"""Request/response schemas for the ``/admin/policies`` routes."""

from __future__ import annotations

from pydantic import BaseModel


class PolicySummary(BaseModel):
    """One ingested policy document, as surfaced back to an admin."""

    source: str
    chunk_count: int


class PolicyListResponse(BaseModel):
    """Response for ``GET /admin/policies``."""

    policies: list[PolicySummary]


class PolicyUploadResponse(BaseModel):
    """Response for ``POST /admin/policies``."""

    source: str
    chunks_ingested: int
    replaced: bool
