"""``/admin`` HTTP routes: dependency health, cache administration, and policy ingestion."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Query, UploadFile
from pydantic import BaseModel, Field

from app.api.deps import get_admin_controller
from app.api.security import require_admin
from app.controllers.admin_controller import AdminController
from app.models.auth_user import AuthenticatedUser
from app.schemas.cache import CacheClearResponse, CacheStatsResponse
from app.schemas.health import HealthCheckResponse
from app.schemas.policy import (
    PendingPolicyApprovalsResponse,
    PolicyListResponse,
    PolicyUploadResponse,
)
from app.schemas.security import SecurityEventsResponse

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(controller: AdminController = Depends(get_admin_controller),
) -> HealthCheckResponse:
    return await controller.health()


@router.get("/cache/stats", response_model=CacheStatsResponse)
def cache_stats(
    _: AuthenticatedUser = Depends(require_admin),
    controller: AdminController = Depends(get_admin_controller),
) -> CacheStatsResponse:
    """Return per-cache hit/miss/set counts. Admin-only."""
    return controller.cache_stats()


@router.post("/cache/clear", response_model=CacheClearResponse)
def cache_clear(
    _: AuthenticatedUser = Depends(require_admin),
    controller: AdminController = Depends(get_admin_controller),
) -> CacheClearResponse:
    """Clear all caches (Redis + counters). Admin-only."""
    return controller.cache_clear()


@router.get("/policies", response_model=PolicyListResponse)
def list_policies(
    _: AuthenticatedUser = Depends(require_admin),
    controller: AdminController = Depends(get_admin_controller),
) -> PolicyListResponse:
    """List every policy document currently ingested. Admin-only."""
    return controller.list_policies()


@router.post("/policies", response_model=PolicyUploadResponse)
def upload_policy(
    file: UploadFile = File(...),
    user: AuthenticatedUser = Depends(require_admin),
    controller: AdminController = Depends(get_admin_controller),
) -> PolicyUploadResponse:
    """Chunk, embed, scan, and quarantine an uploaded policy document.
    Admin-only. The document is NOT searchable yet - it becomes live only
    after a separate approval (``POST /admin/policies/{source}/approve``),
    once its content has passed the guardrails scan
    (``status="scan_passed"`` in the response).

    A plain (non-``async``) route, like the rest of this file's write
    endpoints: FastAPI runs it in its threadpool, so the blocking
    Docling/embedding/scanning work in
    ``PolicyIngestionSecurityService.submit`` doesn't stall the event loop.
    """
    content = file.file.read()
    return controller.upload_policy(file.filename or "", content, user.username)


@router.get("/policies/pending", response_model=PendingPolicyApprovalsResponse)
def pending_policies(
    _: AuthenticatedUser = Depends(require_admin),
    controller: AdminController = Depends(get_admin_controller),
) -> PendingPolicyApprovalsResponse:
    """List uploads awaiting approval (``scan_passed``, not yet live). Admin-only."""
    return controller.pending_policies()


class _RejectPolicyRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=500)


@router.post("/policies/{source}/approve", response_model=PolicyUploadResponse)
def approve_policy(
    source: str,
    user: AuthenticatedUser = Depends(require_admin),
    controller: AdminController = Depends(get_admin_controller),
) -> PolicyUploadResponse:
    """Publish a scan-passed upload: makes it searchable, bumps
    ``corpus_version``, and invalidates the RAG-answer caches. Admin-only.
    """
    return controller.approve_policy(source, user.username)


@router.post("/policies/{source}/reject", response_model=PolicyUploadResponse)
def reject_policy(
    source: str,
    payload: _RejectPolicyRequest,
    user: AuthenticatedUser = Depends(require_admin),
    controller: AdminController = Depends(get_admin_controller),
) -> PolicyUploadResponse:
    """Reject a quarantined upload, deleting its (never-live) chunks. Admin-only."""
    return controller.reject_policy(source, user.username, payload.reason)


@router.get("/security-events", response_model=SecurityEventsResponse)
def security_events(
    limit: int = Query(default=50, gt=0, le=200),
    _: AuthenticatedUser = Depends(require_admin),
    controller: AdminController = Depends(get_admin_controller),
) -> SecurityEventsResponse:
    """Most recent sanitized guardrail-decision audit rows (input/context/
    output/tool/ingestion BLOCK/REDACT decisions, plus monitor-mode
    "would have blocked" entries), newest first. Never contains raw
    scanned text or matched content. Admin-only."""
    return controller.security_events(limit)
