"""``/admin`` HTTP routes: dependency health, cache administration, and policy ingestion."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.deps import get_admin_controller
from app.api.security import require_admin
from app.controllers.admin_controller import AdminController
from app.models.auth_user import AuthenticatedUser
from app.schemas.cache import CacheClearResponse, CacheStatsResponse
from app.schemas.health import HealthCheckResponse
from app.schemas.policy import PolicyListResponse, PolicyUploadResponse

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
    _: AuthenticatedUser = Depends(require_admin),
    controller: AdminController = Depends(get_admin_controller),
) -> PolicyUploadResponse:
    """Chunk, embed, and store an uploaded policy document. Admin-only.

    A plain (non-``async``) route, like the rest of this file's write
    endpoints: FastAPI runs it in its threadpool, so the blocking
    Docling/embedding work in ``PolicyIngestionService.ingest`` doesn't
    stall the event loop.
    """
    content = file.file.read()
    return controller.upload_policy(file.filename or "", content)
