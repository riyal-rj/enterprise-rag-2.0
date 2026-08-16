"""RAG Operations controller: read-side status/metrics aggregation and
write-side config mutation for the admin-only RAG Operations panel.

Every mutating method both persists to the centralized config store
(``RagOpsRepository`` - Postgres-backed, survives a restart) and pushes the
new value into the live singletons (``RAGService``, ``DynamicReranker``,
``SemanticQueryCache``) so the change takes effect on the very next request
*on the worker that handled the admin request* with no process restart or
settings reload - see ``app.rag_services.dynamic_reranker``'s module
docstring for why mutating those singletons in place (rather than
rebuilding them from a config snapshot) is the mechanism that makes that
possible.

Under multiple Uvicorn/Gunicorn workers, every *other* worker's singletons
are untouched by that push - they keep serving whatever config they last
observed until they happen to handle an admin request themselves (or
restart). ``app.api.rag_ops_sync.RagOpsConfigPoller`` closes that gap: each
worker runs its own background task that periodically re-reads the config
row and applies it via :func:`apply_rag_ops_config` below - the same
function this controller's ``_apply`` uses - so a config change (in
particular the emergency kill switch) reaches every worker within one poll
interval instead of only the worker an admin happened to hit.
"""

from __future__ import annotations

from app.core.exceptions import InvalidRagOpsConfigError
from app.models.rag_ops import RagOpsConfig
from app.rag_services.dynamic_reranker import DynamicReranker
from app.rag_services.rag_service import RAGService
from app.repositories.rag_ops_repository import RagOpsRepository
from app.repositories.semantic_cache_repository import SemanticQueryCache
from app.schemas.rag_ops import (
    AuditLogEntryResponse,
    AuditLogResponse,
    EmergencyDisableRequest,
    EmergencyEnableRequest,
    RagOpsConfigUpdateRequest,
    RagOpsStatusResponse,
    RerankMetrics,
    SemanticCacheMetrics,
)
from app.services.rag_metrics_service import RagMetricsService


def apply_rag_ops_config(
    config: RagOpsConfig,
    *,
    rag_service: RAGService,
    reranker: DynamicReranker,
    semantic_query_cache: SemanticQueryCache,
) -> None:
    """Push ``config`` into every live singleton that needs to observe it
    immediately. Emergency disable forces reranking and semantic caching
    off regardless of their individually stored "enabled" flags - it's
    meant to revert to the safe baseline retrieve-then-generate pipeline,
    not just gate the reranker.

    Shared by :meth:`RagOpsController._apply` (the worker that handled the
    admin request) and ``app.api.rag_ops_sync.RagOpsConfigPoller`` (every
    other worker, on its next poll tick) so both apply a config change
    identically - see module docstring.
    """
    rag_service.set_reranking_enabled(config.reranking_enabled and not config.emergency_disabled)
    rag_service.set_semantic_cache_enabled(
        config.semantic_cache_enabled and not config.emergency_disabled
    )
    reranker.set_backend(config.reranker_backend)  # type: ignore[arg-type]
    reranker.set_rollout_percentage(config.reranker_rollout_percentage)
    reranker.set_emergency_disabled(config.emergency_disabled)
    semantic_query_cache.set_similarity_threshold(config.semantic_cache_threshold)


class RagOpsController:
    """Orchestrates :class:`RagOpsRepository` + live-singleton pushes for
    the ``/admin/rag-ops`` routes."""

    def __init__(
        self,
        repository: RagOpsRepository,
        rag_service: RAGService,
        reranker: DynamicReranker,
        semantic_query_cache: SemanticQueryCache,
        metrics: RagMetricsService,
    ) -> None:
        self._repository = repository
        self._rag_service = rag_service
        self._reranker = reranker
        self._semantic_query_cache = semantic_query_cache
        self._metrics = metrics

    def status(self) -> RagOpsStatusResponse:
        return self._to_status(self._repository.get_config())

    def update_config(self, actor: str, payload: RagOpsConfigUpdateRequest) -> RagOpsStatusResponse:
        if payload.reranker_backend == "voyage" and not self._reranker.has_voyage_backend:
            raise InvalidRagOpsConfigError(
                "Cannot switch to the Voyage reranker backend: no VOYAGE_API_KEY is "
                "configured for this deployment."
            )

        config = self._repository.update_config(
            actor=actor,
            reason=payload.reason,
            reranking_enabled=payload.reranking_enabled,
            reranker_backend=payload.reranker_backend,
            reranker_rollout_percentage=payload.reranker_rollout_percentage,
            semantic_cache_enabled=payload.semantic_cache_enabled,
            semantic_cache_threshold=payload.semantic_cache_threshold,
        )
        self._apply(config)
        return self._to_status(config)

    def emergency_disable(self, actor: str, payload: EmergencyDisableRequest) -> RagOpsStatusResponse:
        config = self._repository.set_emergency_disabled(
            actor=actor, disabled=True, reason=payload.reason
        )
        self._apply(config)
        return self._to_status(config)

    def emergency_enable(self, actor: str, payload: EmergencyEnableRequest) -> RagOpsStatusResponse:
        config = self._repository.set_emergency_disabled(
            actor=actor, disabled=False, reason=payload.reason
        )
        self._apply(config)
        return self._to_status(config)

    def audit_log(self, limit: int) -> AuditLogResponse:
        entries = self._repository.list_audit(limit)
        return AuditLogResponse(
            items=[
                AuditLogEntryResponse(
                    id=entry.id,
                    actor=entry.actor,
                    action=entry.action,
                    changes=entry.changes,
                    reason=entry.reason,
                    created_at=entry.created_at,
                )
                for entry in entries
            ]
        )

    def _apply(self, config: RagOpsConfig) -> None:
        """Push the newly-persisted config into this worker's live
        singletons immediately (see module docstring for the cross-worker
        story)."""
        apply_rag_ops_config(
            config,
            rag_service=self._rag_service,
            reranker=self._reranker,
            semantic_query_cache=self._semantic_query_cache,
        )

    def _to_status(self, config: RagOpsConfig) -> RagOpsStatusResponse:
        rerank_snapshot = self._metrics.rerank_stats()
        semantic_snapshot = self._metrics.semantic_cache_stats()
        return RagOpsStatusResponse(
            reranking_enabled=config.reranking_enabled,
            reranker_backend=config.reranker_backend,  # type: ignore[arg-type]
            reranker_rollout_percentage=config.reranker_rollout_percentage,
            reranker_metrics=RerankMetrics(
                sample_count=rerank_snapshot.sample_count,
                p50_latency_ms=rerank_snapshot.p50_latency_ms,
                p95_latency_ms=rerank_snapshot.p95_latency_ms,
                fallback_rate=rerank_snapshot.fallback_rate,
                voyage_tokens_total=rerank_snapshot.voyage_tokens_total,
            ),
            semantic_cache_enabled=config.semantic_cache_enabled,
            semantic_cache_threshold=config.semantic_cache_threshold,
            semantic_cache_metrics=SemanticCacheMetrics(
                lookups=semantic_snapshot.lookups,
                hits=semantic_snapshot.hits,
                hit_rate=semantic_snapshot.hit_rate,
            ),
            emergency_disabled=config.emergency_disabled,
            emergency_disabled_reason=config.emergency_disabled_reason,
            emergency_disabled_at=config.emergency_disabled_at,
            emergency_disabled_by=config.emergency_disabled_by,
            corpus_version=config.corpus_version,
            last_cache_invalidated_at=config.last_cache_invalidated_at,
            updated_at=config.updated_at,
            updated_by=config.updated_by,
        )
