"""Persistence for the centralized RAG operations config (feature-flag/
config service) and its audit trail - backs the admin-only RAG Operations
panel (see ``app.controllers.rag_ops_controller``).

A single mutable row (id=1) rather than one row per flag: every field here
is read together on every panel load and written together by a handful of
admin actions, so a wider EAV-style flags table would buy nothing, and a
fixed-column row keeps every write's audit diff a plain attribute
comparison (see ``_diff``).
"""

from __future__ import annotations

from typing import Any, Protocol

from psycopg2.extensions import cursor as PgCursor
from psycopg2.extras import Json

from app.core.db import PostgresConnectionPool
from app.models.rag_ops import RagOpsAuditEntry, RagOpsConfig

_CONFIG_COLUMNS = (
    "id, reranking_enabled, reranker_backend, reranker_rollout_percentage, "
    "semantic_cache_enabled, semantic_cache_threshold, hyde_enabled, "
    "hyde_rollout_percentage, emergency_disabled, "
    "emergency_disabled_reason, emergency_disabled_at, emergency_disabled_by, "
    "corpus_version, last_cache_invalidated_at, updated_at, updated_by"
)

_DIFF_FIELDS = (
    "reranking_enabled",
    "reranker_backend",
    "reranker_rollout_percentage",
    "semantic_cache_enabled",
    "semantic_cache_threshold",
    "hyde_enabled",
    "hyde_rollout_percentage",
)


class RagOpsRepository(Protocol):
    """Persistence contract for the RAG Operations panel's config + audit trail."""

    def get_config(self) -> RagOpsConfig: ...

    def update_config(
        self,
        *,
        actor: str,
        reason: str | None,
        reranking_enabled: bool | None = None,
        reranker_backend: str | None = None,
        reranker_rollout_percentage: int | None = None,
        semantic_cache_enabled: bool | None = None,
        semantic_cache_threshold: float | None = None,
        hyde_enabled: bool | None = None,
        hyde_rollout_percentage: int | None = None,
    ) -> RagOpsConfig:
        """Partial update: ``None`` fields are left unchanged. Writes an
        audit row iff at least one field actually changed value."""
        ...

    def set_emergency_disabled(
        self, *, actor: str, disabled: bool, reason: str | None
    ) -> RagOpsConfig: ...

    def bump_corpus_version(self, *, actor: str, source: str) -> RagOpsConfig: ...

    def record_cache_invalidation(self) -> RagOpsConfig:
        """Timestamp-only - deliberately not audited (see module docstring
        on ``PostgresRagOpsRepository.record_cache_invalidation``)."""
        ...

    def list_audit(self, limit: int) -> list[RagOpsAuditEntry]: ...


class PostgresRagOpsRepository:
    """Postgres-backed :class:`RagOpsRepository`."""

    def __init__(self, pool: PostgresConnectionPool) -> None:
        self._pool = pool

    def get_config(self) -> RagOpsConfig:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT {_CONFIG_COLUMNS} FROM rag_ops_config WHERE id = 1")
            row = cur.fetchone()
        if row is None:
            raise RuntimeError(
                "rag_ops_config has no row - run migrations "
                "(app/seed/migrations/005_create_rag_ops_config.sql)"
            )
        return _row_to_config(row)

    def update_config(
        self,
        *,
        actor: str,
        reason: str | None,
        reranking_enabled: bool | None = None,
        reranker_backend: str | None = None,
        reranker_rollout_percentage: int | None = None,
        semantic_cache_enabled: bool | None = None,
        semantic_cache_threshold: float | None = None,
        hyde_enabled: bool | None = None,
        hyde_rollout_percentage: int | None = None,
    ) -> RagOpsConfig:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                # FOR UPDATE: the audit diff below compares this snapshot
                # against the post-UPDATE row, so a concurrent admin write
                # racing the same singleton row must be serialized, not
                # just the column write itself.
                cur.execute(f"SELECT {_CONFIG_COLUMNS} FROM rag_ops_config WHERE id = 1 FOR UPDATE")
                old_row = cur.fetchone()
                if old_row is None:
                    raise RuntimeError("rag_ops_config has no row - run migrations")
                old = _row_to_config(old_row)

                cur.execute(
                    f"""
                    UPDATE rag_ops_config SET
                        reranking_enabled = COALESCE(%s, reranking_enabled),
                        reranker_backend = COALESCE(%s, reranker_backend),
                        reranker_rollout_percentage = COALESCE(%s, reranker_rollout_percentage),
                        semantic_cache_enabled = COALESCE(%s, semantic_cache_enabled),
                        semantic_cache_threshold = COALESCE(%s, semantic_cache_threshold),
                        hyde_enabled = COALESCE(%s, hyde_enabled),
                        hyde_rollout_percentage = COALESCE(%s, hyde_rollout_percentage),
                        updated_at = now(),
                        updated_by = %s
                    WHERE id = 1
                    RETURNING {_CONFIG_COLUMNS}
                    """,
                    (
                        reranking_enabled,
                        reranker_backend,
                        reranker_rollout_percentage,
                        semantic_cache_enabled,
                        semantic_cache_threshold,
                        hyde_enabled,
                        hyde_rollout_percentage,
                        actor,
                    ),
                )
                new = _row_to_config(cur.fetchone())

                changes = _diff(old, new)
                if changes:
                    self._insert_audit(cur, actor, "config_update", changes, reason)
            conn.commit()
        return new

    def set_emergency_disabled(
        self, *, actor: str, disabled: bool, reason: str | None
    ) -> RagOpsConfig:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE rag_ops_config SET
                        emergency_disabled = %s,
                        emergency_disabled_reason = %s,
                        emergency_disabled_at = CASE WHEN %s THEN now() ELSE NULL END,
                        emergency_disabled_by = CASE WHEN %s THEN %s ELSE NULL END,
                        updated_at = now(),
                        updated_by = %s
                    WHERE id = 1
                    RETURNING {_CONFIG_COLUMNS}
                    """,
                    (disabled, reason, disabled, disabled, actor, actor),
                )
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError("rag_ops_config has no row - run migrations")
                new = _row_to_config(row)
                self._insert_audit(
                    cur,
                    actor,
                    "emergency_disable" if disabled else "emergency_enable",
                    {"emergency_disabled": {"new": disabled}},
                    reason,
                )
            conn.commit()
        return new

    def bump_corpus_version(self, *, actor: str, source: str) -> RagOpsConfig:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE rag_ops_config
                    SET corpus_version = corpus_version + 1, updated_at = now(), updated_by = %s
                    WHERE id = 1
                    RETURNING {_CONFIG_COLUMNS}
                    """,
                    (actor,),
                )
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError("rag_ops_config has no row - run migrations")
                new = _row_to_config(row)
                self._insert_audit(
                    cur,
                    actor,
                    "corpus_version_bump",
                    {"source": source, "corpus_version": {"new": new.corpus_version}},
                    None,
                )
            conn.commit()
        return new

    def record_cache_invalidation(self) -> RagOpsConfig:
        """Bumps ``last_cache_invalidated_at`` only - not audited. Cache
        clears are a frequent, low-stakes admin action (unlike a config
        change or emergency action) and already have their own visible
        timestamp field in the panel; logging one every time would just
        drown out the audit trail's actually-interesting entries."""
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE rag_ops_config SET last_cache_invalidated_at = now()
                WHERE id = 1
                RETURNING {_CONFIG_COLUMNS}
                """
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("rag_ops_config has no row - run migrations")
            new = _row_to_config(row)
            conn.commit()
        return new

    def list_audit(self, limit: int) -> list[RagOpsAuditEntry]:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, actor, action, changes, reason, created_at "
                "FROM rag_ops_audit_log ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
        return [
            RagOpsAuditEntry(
                id=row[0],
                actor=row[1],
                action=row[2],
                changes=row[3] or {},
                reason=row[4],
                created_at=row[5],
            )
            for row in rows
        ]

    def _insert_audit(
        self, cur: PgCursor, actor: str, action: str, changes: dict[str, Any], reason: str | None
    ) -> None:
        cur.execute(
            "INSERT INTO rag_ops_audit_log (actor, action, changes, reason) "
            "VALUES (%s, %s, %s, %s)",
            (actor, action, Json(changes), reason),
        )


def _diff(old: RagOpsConfig, new: RagOpsConfig) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    for field_name in _DIFF_FIELDS:
        old_value = getattr(old, field_name)
        new_value = getattr(new, field_name)
        if old_value != new_value:
            changes[field_name] = {"old": old_value, "new": new_value}
    return changes


def _row_to_config(row: tuple[Any, ...]) -> RagOpsConfig:
    return RagOpsConfig(
        reranking_enabled=row[1],
        reranker_backend=row[2],
        reranker_rollout_percentage=row[3],
        semantic_cache_enabled=row[4],
        semantic_cache_threshold=row[5],
        hyde_enabled=row[6],
        hyde_rollout_percentage=row[7],
        emergency_disabled=row[8],
        emergency_disabled_reason=row[9],
        emergency_disabled_at=row[10],
        emergency_disabled_by=row[11],
        corpus_version=row[12],
        last_cache_invalidated_at=row[13],
        updated_at=row[14],
        updated_by=row[15],
    )
