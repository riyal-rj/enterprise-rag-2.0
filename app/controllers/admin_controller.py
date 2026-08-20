"""Admin controller: health checks, query-cache administration, and policy ingestion."""

from __future__ import annotations

from app.repositories.rag_ops_repository import RagOpsRepository
from app.repositories.security_events_repository import SecurityEventsRepository
from app.repositories.semantic_cache_repository import SemanticQueryCache
from app.schemas.cache import CacheClearResponse, CacheStatsResponse, CacheTierStats
from app.schemas.health import HealthCheckResponse
from app.schemas.policy import (
    DocumentSecurityStateResponse,
    PendingPolicyApprovalsResponse,
    PolicyListResponse,
    PolicyUploadResponse,
)
from app.schemas.security import SecurityEventResponse, SecurityEventsResponse
from app.services.corpus_cache_invalidation_service import invalidate_corpus_caches
from app.services.health_checks import HealthCheckService
from app.services.policy_ingestion_security_service import PolicyIngestionSecurityService
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
        self,
        health_check_service: HealthCheckService,
        query_cache: QueryCacheService,
        policy_ingestion_service: PolicyIngestionSecurityService,
        semantic_query_cache: SemanticQueryCache | None = None,
        rag_ops_repository: RagOpsRepository | None = None,
        security_events_repository: SecurityEventsRepository | None = None,
    ) -> None:
        self._health_check_service = health_check_service
        self._query_cache = query_cache
        self._policy_ingestion_service = policy_ingestion_service
        self._semantic_query_cache = semantic_query_cache
        self._rag_ops_repository = rag_ops_repository
        self._security_events_repository = security_events_repository

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
        if self._rag_ops_repository is not None:
            self._rag_ops_repository.record_cache_invalidation()
        return CacheClearResponse(status="ok", cleared=cleared)

    def list_policies(self) -> PolicyListResponse:
        return self._policy_ingestion_service.list_policies()

    def pending_policies(self) -> PendingPolicyApprovalsResponse:
        items = [
            DocumentSecurityStateResponse(
                id=state.id,
                source=state.source,
                status=state.status,
                uploaded_by=state.uploaded_by,
                chunk_count=state.chunk_count,
                rejected_reason=state.rejected_reason,
            )
            for state in self._policy_ingestion_service.list_pending()
        ]
        return PendingPolicyApprovalsResponse(items=items)

    def upload_policy(self, filename: str, content: bytes, actor: str) -> PolicyUploadResponse:
        """Quarantine, scan, and store ``filename`` - see
        ``app.services.policy_ingestion_security_service``. The document is
        NOT live/searchable yet: cache invalidation only happens on
        :meth:`approve_policy`, since a quarantined document was never
        served and has nothing stale to invalidate.
        """
        return self._policy_ingestion_service.submit(filename, content, actor)

    def approve_policy(self, source: str, actor: str) -> PolicyUploadResponse:
        """Publish a scan-passed document and, since it just became live,
        bump ``corpus_version`` and invalidate the RAG-answer caches (see
        ``app.services.corpus_cache_invalidation_service`` - targeted, not
        the full ``cache_clear()`` wipe: an unrelated embedding/SQL/intent
        cold start isn't warranted by a policy approval).

        Without this, a replaced policy's stale pre-upload answers would
        keep being served from Redis (and from semantic pointers resolving
        back to those same Redis entries) until they happen to expire on
        their own TTL - the operations panel would show the new corpus
        version while the runtime kept answering from the old one.
        """
        state = self._policy_ingestion_service.approve(source, actor)
        if self._rag_ops_repository is not None:
            invalidate_corpus_caches(
                rag_ops_repository=self._rag_ops_repository,
                query_cache=self._query_cache,
                semantic_query_cache=self._semantic_query_cache,
                actor=actor,
                source=source,
            )
        return PolicyUploadResponse(
            source=state.source,
            chunks_ingested=state.chunk_count or 0,
            replaced=False,
            status=state.status,
        )

    def reject_policy(self, source: str, actor: str, reason: str) -> PolicyUploadResponse:
        state = self._policy_ingestion_service.reject(source, actor, reason)
        return PolicyUploadResponse(
            source=state.source,
            chunks_ingested=state.chunk_count or 0,
            replaced=False,
            status=state.status,
        )

    def security_events(self, limit: int) -> SecurityEventsResponse:
        """Most recent sanitized guardrail-decision audit rows (see
        ``app.guardrails``), newest first. Empty if no repository was
        wired in (e.g. a test/eval construction site)."""
        if self._security_events_repository is None:
            return SecurityEventsResponse(items=[])
        events = self._security_events_repository.list_recent(limit)
        return SecurityEventsResponse(
            items=[
                SecurityEventResponse(
                    id=event.id,
                    actor=event.actor,
                    action=event.action,
                    stage=event.stage,
                    category=event.category,
                    mode=event.mode,
                    changes=event.changes,
                    reason=event.reason,
                    created_at=event.created_at,
                )
                for event in events
            ]
        )
