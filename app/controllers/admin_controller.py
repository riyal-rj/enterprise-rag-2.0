"""Admin controller: health checks, query-cache administration, and policy ingestion."""

from __future__ import annotations

from app.repositories.semantic_cache_repository import SemanticQueryCache
from app.schemas.cache import CacheClearResponse, CacheStatsResponse, CacheTierStats
from app.schemas.health import HealthCheckResponse
from app.schemas.policy import PolicyListResponse, PolicyUploadResponse
from app.services.health_checks import HealthCheckService
from app.services.policy_ingestion_service import PolicyIngestionService
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

    def __init__(self,
                 health_check_service: HealthCheckService,
                 query_cache: QueryCacheService,
                 policy_ingestion_service: PolicyIngestionService,
                 semantic_query_cache: SemanticQueryCache | None = None) -> None:
        self._health_check_service = health_check_service
        self._query_cache = query_cache
        self._policy_ingestion_service = policy_ingestion_service
        self._semantic_query_cache = semantic_query_cache

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
        if self._semantic_query_cache is not None:
            cleared["semantic"] = self._semantic_query_cache.clear()
        return CacheClearResponse(status="ok", cleared=cleared)

    def list_policies(self) -> PolicyListResponse:
        return self._policy_ingestion_service.list_policies()

    def upload_policy(self, filename: str, content: bytes) -> PolicyUploadResponse:
        return self._policy_ingestion_service.ingest(filename, content)
