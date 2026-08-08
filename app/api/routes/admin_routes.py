"""``/admin`` HTTP routes: dependency health and cache administration."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_admin_controller
from app.api.security import require_admin
from app.controllers.admin_controller import AdminController
from app.models.auth_user import AuthenticatedUser
from app.schemas.cache import CacheClearResponse, CacheStatsResponse
from app.schemas.health import HealthCheckResponse

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
