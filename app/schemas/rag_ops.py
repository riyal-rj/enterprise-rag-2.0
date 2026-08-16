"""Request/response schemas for the ``/admin/rag-ops`` (RAG Operations panel) routes."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

RerankerBackend = Literal["local", "voyage"]


class RerankMetrics(BaseModel):
    """Reranker performance over the last N *attempted* reranks (a rolling
    in-memory window - see ``app.services.rag_metrics_service``). ``None``
    latencies mean no rerank has been attempted yet since the process
    started."""

    sample_count: int
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    fallback_rate: float
    voyage_tokens_total: int


class SemanticCacheMetrics(BaseModel):
    lookups: int
    hits: int
    hit_rate: float


class RagOpsStatusResponse(BaseModel):
    """Everything the RAG Operations panel displays: live config, rolled-up
    metrics, and emergency-disable state."""

    reranking_enabled: bool
    reranker_backend: RerankerBackend
    reranker_rollout_percentage: int
    reranker_metrics: RerankMetrics

    semantic_cache_enabled: bool
    semantic_cache_threshold: float
    semantic_cache_metrics: SemanticCacheMetrics

    emergency_disabled: bool
    emergency_disabled_reason: str | None
    emergency_disabled_at: datetime | None
    emergency_disabled_by: str | None

    corpus_version: int
    last_cache_invalidated_at: datetime | None

    updated_at: datetime
    updated_by: str | None


class RagOpsConfigUpdateRequest(BaseModel):
    """Partial update for ``PATCH /admin/rag-ops/config``: omitted/``null``
    fields are left unchanged."""

    reranking_enabled: bool | None = None
    reranker_backend: RerankerBackend | None = None
    reranker_rollout_percentage: int | None = Field(default=None, ge=0, le=100)
    semantic_cache_enabled: bool | None = None
    semantic_cache_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str | None = Field(default=None, max_length=500)


class EmergencyDisableRequest(BaseModel):
    """Body for ``POST /admin/rag-ops/emergency-disable``.

    ``confirm`` must be sent as literal ``true`` - this is a server-side
    backstop for the confirmation the panel's UI already requires, not a
    replacement for it.
    """

    reason: str = Field(min_length=1, max_length=500)
    confirm: Literal[True]


class EmergencyEnableRequest(BaseModel):
    reason: str | None = Field(default=None, max_length=500)


class AuditLogEntryResponse(BaseModel):
    id: int
    actor: str
    action: str
    changes: dict[str, Any]
    reason: str | None
    created_at: datetime


class AuditLogResponse(BaseModel):
    items: list[AuditLogEntryResponse]
