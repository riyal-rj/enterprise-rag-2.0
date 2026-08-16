"""``/admin/rag-ops`` HTTP routes: the RAG Operations panel's backend.

All admin-only. See ``app.controllers.rag_ops_controller`` for the
mutation -> live-singleton-push mechanism that makes config changes
effective without a process restart.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_rag_ops_controller
from app.api.security import require_admin
from app.controllers.rag_ops_controller import RagOpsController
from app.models.auth_user import AuthenticatedUser
from app.schemas.rag_ops import (
    AuditLogResponse,
    EmergencyDisableRequest,
    EmergencyEnableRequest,
    RagOpsConfigUpdateRequest,
    RagOpsStatusResponse,
)

router = APIRouter(prefix="/admin/rag-ops", tags=["admin", "rag-ops"])


@router.get("/status", response_model=RagOpsStatusResponse)
def status(
    _: AuthenticatedUser = Depends(require_admin),
    controller: RagOpsController = Depends(get_rag_ops_controller),
) -> RagOpsStatusResponse:
    """Current reranking/semantic-cache config, live metrics, and emergency state. Admin-only."""
    return controller.status()


@router.patch("/config", response_model=RagOpsStatusResponse)
def update_config(
    payload: RagOpsConfigUpdateRequest,
    user: AuthenticatedUser = Depends(require_admin),
    controller: RagOpsController = Depends(get_rag_ops_controller),
) -> RagOpsStatusResponse:
    """Partially update the centralized RAG config. Takes effect immediately, on
    every server process, with no restart. Admin-only."""
    return controller.update_config(user.username, payload)


@router.post("/emergency-disable", response_model=RagOpsStatusResponse)
def emergency_disable(
    payload: EmergencyDisableRequest,
    user: AuthenticatedUser = Depends(require_admin),
    controller: RagOpsController = Depends(get_rag_ops_controller),
) -> RagOpsStatusResponse:
    """Kill switch: force reranking and semantic caching off immediately,
    falling back to the baseline retrieve-then-generate pipeline. Requires
    an explicit confirmation and a reason. Admin-only."""
    return controller.emergency_disable(user.username, payload)


@router.post("/emergency-enable", response_model=RagOpsStatusResponse)
def emergency_enable(
    payload: EmergencyEnableRequest,
    user: AuthenticatedUser = Depends(require_admin),
    controller: RagOpsController = Depends(get_rag_ops_controller),
) -> RagOpsStatusResponse:
    """Clear the emergency kill switch, restoring the previously configured
    reranking/semantic-cache state. Admin-only."""
    return controller.emergency_enable(user.username, payload)


@router.get("/audit", response_model=AuditLogResponse)
def audit_log(
    limit: int = Query(default=50, gt=0, le=200),
    _: AuthenticatedUser = Depends(require_admin),
    controller: RagOpsController = Depends(get_rag_ops_controller),
) -> AuditLogResponse:
    """Most recent config mutations, emergency actions, and corpus version
    bumps, newest first. Admin-only."""
    return controller.audit_log(limit)
