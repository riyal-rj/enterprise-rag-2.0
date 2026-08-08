"""Admin controller: health checks and query-cache administration."""

from __future__ import annotations

from app.schemas.cache import CacheClearResponse, CacheStatsResponse, CacheTierStats
from app.schemas.health import HealthCheckResponse
from app.services.health_checks import HealthCheckService
from app.services.query_cache_service import CacheTier, QueryCacheService

_TIER_TO_RESPONSE_FIELD: dict[CacheTier, str] = {
    CacheTier.EMBEDDING: "embedding",
    CacheTier.RAG_ANSWER: "rag",
    CacheTier.SQL_GEN: "sql_gen",
    CacheTier.SQL_RESULT: "sql_result",
    CacheTier.INTENT: "intent_router",
}


class AdminController:
    """Orchestrates health checks and cache admin calls for ``/admin`` routes."""

    def __init__(
        self, health_check_service: HealthCheckService, query_cache: QueryCacheService
    ) -> None:
        self._health_check_service = health_check_service
        self._query_cache = query_cache

    async def health(self) -> HealthCheckResponse:
        results = await self._health_check_service.run()
        status = "ok" if all(results.values()) else "degraded"
        return HealthCheckResponse(status=status, **results)

    def cache_stats(self) -> CacheStatsResponse:
        raw = self._query_cache.stats()
        fields = {
            field_name: CacheTierStats(**raw[tier.value])
            for tier, field_name in _TIER_TO_RESPONSE_FIELD.items()
        }
        return CacheStatsResponse(**fields)

    def cache_clear(self) -> CacheClearResponse:
        cleared = self._query_cache.clear()
        return CacheClearResponse(status="ok", cleared=cleared)
