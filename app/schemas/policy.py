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
    """Response for ``POST /admin/policies``.

    ``status`` reflects the quarantine workflow's state immediately after
    upload (``"pending_scan"``/``"scan_passed"``/``"scan_failed"`` - never
    ``"active"`` here, since activation only happens via a separate admin
    approval, see ``POST /admin/policies/{source}/approve``) - tells the
    caller the document is not yet live/searchable.
    """

    source: str
    chunks_ingested: int
    replaced: bool
    status: str = "active"


class DocumentSecurityStateResponse(BaseModel):
    """One row of the ingestion-quarantine lifecycle, as surfaced to an admin."""

    id: int
    source: str
    status: str
    uploaded_by: str
    chunk_count: int | None
    rejected_reason: str | None


class PendingPolicyApprovalsResponse(BaseModel):
    """Response for ``GET /admin/policies/pending``."""

    items: list[DocumentSecurityStateResponse]
