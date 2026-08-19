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
    hyde_enabled: bool
    hyde_rollout_percentage: int
    crag_enabled: bool
    crag_rollout_percentage: int
    crag_web_enabled: bool
    emergency_disabled: bool
    emergency_disabled_reason: str | None
    emergency_disabled_at: datetime | None
    emergency_disabled_by: str | None
    corpus_version: int
    last_cache_invalidated_at: datetime | None
    updated_at: datetime
    updated_by: str | None
    # Observe-only rollout stage - see app.rag_services.crag.dynamic_corrective_retriever
    # and RAGService.answer's "shadow" cohort handling. Defaulted so every
    # pre-existing construction site (tests, PostgresRagOpsRepository rows
    # from before this column existed) keeps working unchanged.
    crag_shadow_enabled: bool = False
    # Self-reflective answer critique/revision stage - see
    # app.rag_services.reflection.dynamic_self_reflection. Defaulted for the
    # same reason as crag_shadow_enabled above.
    self_reflective_enabled: bool = False
    self_reflective_rollout_percentage: int = 0
    # Observe-only rollout stage for self-reflection, same role as
    # crag_shadow_enabled above - see
    # app.rag_services.reflection.dynamic_self_reflection and
    # RAGService.answer's self-reflection "shadow" cohort handling.
    self_reflective_shadow_enabled: bool = False
    # Revision-only canary gate: the bounded additional-retrieval step is
    # unavailable (RETRIEVE_MORE degrades to REVISE) until this is turned on
    # separately from self_reflective_enabled itself - see
    # app.rag_services.reflection.reflection_engine.
    self_reflective_retrieval_enabled: bool = False
    # Text-to-SQL routing - see app.query_orchestration.query_orchestrator.
    # sql_proposal_only is DB-constrained to always be TRUE (migration 011)
    # and re-validated in RagRuntimeConfig.__post_init__ - this release has
    # no automatic-execution path. Defaulted for the same reason as
    # crag_shadow_enabled above.
    sql_enabled: bool = False
    sql_rollout_percentage: int = 0
    sql_proposal_only: bool = True


def validate_crag_state(
    *, crag_enabled: bool, crag_web_enabled: bool, crag_shadow_enabled: bool = False
) -> None:
    """Invariant shared by every write path that can change these fields
    (API request payload, DB row after a concurrent write) - web correction
    and shadow mode must never be enabled while CRAG itself is off."""
    if crag_web_enabled and not crag_enabled:
        raise ValueError("crag_web_enabled requires crag_enabled")
    if crag_shadow_enabled and not crag_enabled:
        raise ValueError("crag_shadow_enabled requires crag_enabled")


def validate_self_reflective_state(
    *,
    self_reflective_enabled: bool,
    self_reflective_shadow_enabled: bool = False,
    self_reflective_retrieval_enabled: bool = False,
) -> None:
    """Invariant shared by every write path that can change these fields -
    mirrors ``validate_crag_state``'s role for CRAG's dependent sub-flags."""
    if self_reflective_shadow_enabled and not self_reflective_enabled:
        raise ValueError("self_reflective_shadow_enabled requires self_reflective_enabled")
    if self_reflective_retrieval_enabled and not self_reflective_enabled:
        raise ValueError("self_reflective_retrieval_enabled requires self_reflective_enabled")


@dataclass(frozen=True)
class RagOpsAuditEntry:
    """One row of the append-only audit trail behind a config mutation."""

    id: int
    actor: str
    action: str
    changes: dict[str, Any]
    reason: str | None
    created_at: datetime
