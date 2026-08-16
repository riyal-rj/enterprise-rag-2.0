"""Persisted RAG operations config + audit trail rows.

Distinct from the API-facing pydantic schemas in ``app.schemas.rag_ops``
(the same data shaped for a response body) and from
``app.core.config.rag_features.RAGFeatureSettings`` (the frozen, env-only
defaults that only seed this table's initial row - see
``app/seed/migrations/005_create_rag_ops_config.sql``). Once that row
exists, it - not the environment - is authoritative for these fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class RagOpsConfig:
    """The RAG Operations panel's live, admin-mutable config (singleton row)."""

    reranking_enabled: bool
    reranker_backend: str
    reranker_rollout_percentage: int
    semantic_cache_enabled: bool
    semantic_cache_threshold: float
    emergency_disabled: bool
    emergency_disabled_reason: str | None
    emergency_disabled_at: datetime | None
    emergency_disabled_by: str | None
    corpus_version: int
    last_cache_invalidated_at: datetime | None
    updated_at: datetime
    updated_by: str | None


@dataclass(frozen=True)
class RagOpsAuditEntry:
    """One row of the append-only audit trail behind a config mutation."""

    id: int
    actor: str
    action: str
    changes: dict[str, Any]
    reason: str | None
    created_at: datetime
