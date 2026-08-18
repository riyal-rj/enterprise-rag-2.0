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


class HyDEMetrics(BaseModel):
    """HyDE performance over the last N *attempted* transforms (a rolling
    in-memory window - see ``app.services.rag_metrics_service``). ``None``
    latencies mean no HyDE attempt has happened yet since the process
    started. ``rollout_bypasses``/``emergency_bypasses`` count requests that
    never reached the delegate at all (sampled out / emergency-disabled),
    distinct from ``fallback_rate``, which is over attempted transforms."""

    sample_count: int
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    fallback_rate: float
    usage_tokens_total: int
    rollout_bypasses: int
    emergency_bypasses: int


class CRAGMetrics(BaseModel):
    """CRAG performance over the last N *attempted* corrections (a rolling
    in-memory window - see ``app.services.rag_metrics_service``). ``None``
    latencies mean no CRAG attempt has happened yet since the process
    started. ``rollout_bypasses``/``emergency_bypasses`` count requests that
    never reached the corrective retriever at all (sampled out /
    emergency-disabled), same distinction as ``HyDEMetrics``."""

    sample_count: int
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    correct_count: int
    ambiguous_count: int
    incorrect_count: int
    fallback_rate: float
    abstention_rate: float
    web_use_rate: float
    usage_tokens_total: int
    rollout_bypasses: int
    emergency_bypasses: int


class CRAGShadowMetrics(BaseModel):
    """Shadow-cohort CRAG performance - never served to a real user, tracked
    separately from ``CRAGMetrics`` so a shadow regression is visible
    without being confused with what traffic is actually experiencing."""

    sample_count: int
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    correct_count: int
    ambiguous_count: int
    incorrect_count: int
    fallback_rate: float
    abstention_rate: float
    web_use_rate: float
    usage_tokens_total: int


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

    hyde_enabled: bool
    hyde_rollout_percentage: int
    hyde_metrics: HyDEMetrics

    crag_enabled: bool
    crag_rollout_percentage: int
    crag_web_enabled: bool
    crag_web_available: bool
    crag_shadow_enabled: bool
    crag_metrics: CRAGMetrics
    crag_shadow_metrics: CRAGShadowMetrics

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
    hyde_enabled: bool | None = None
    hyde_rollout_percentage: int | None = Field(default=None, ge=0, le=100)
    crag_enabled: bool | None = None
    crag_rollout_percentage: int | None = Field(default=None, ge=0, le=100)
    crag_web_enabled: bool | None = None
    crag_shadow_enabled: bool | None = None
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
