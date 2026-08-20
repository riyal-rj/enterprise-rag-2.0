"""Persistence for the policy-upload quarantine workflow's per-document
lifecycle (``pending_scan -> scan_passed/scan_failed -> approved -> active``
or ``rejected``) - see ``app.services.policy_ingestion_security_service``
and ``app/seed/migrations/014_create_document_security_state.sql``.

Same shape as ``app.repositories.rag_ops_repository``'s audit trail
(``RagOpsAuditEntry``/``_insert_audit``), but this table is keyed by
uploaded document, not by an admin actor mutating global config - one row
per upload, never updated-in-place across a re-upload (see the migration's
own docstring).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from psycopg2.extras import Json

from app.core.db import PostgresConnectionPool

_COLUMNS = (
    "id, source, status, uploaded_by, uploaded_at, scan_decision, "
    "scanned_at, approved_by, approved_at, rejected_reason, chunk_count"
)


@dataclass(frozen=True)
class DocumentSecurityState:
    id: int
    source: str
    status: str
    uploaded_by: str
    uploaded_at: datetime
    scan_decision: dict[str, Any] | None
    scanned_at: datetime | None
    approved_by: str | None
    approved_at: datetime | None
    rejected_reason: str | None
    chunk_count: int | None


class DocumentSecurityRepository(Protocol):
    def create_pending(
        self, *, source: str, uploaded_by: str, chunk_count: int
    ) -> DocumentSecurityState: ...

    def record_scan_result(
        self, *, id: int, status: str, scan_decision: dict[str, Any]
    ) -> DocumentSecurityState: ...

    def approve(self, *, id: int, actor: str) -> DocumentSecurityState: ...

    def mark_active(self, *, id: int) -> DocumentSecurityState: ...

    def reject(self, *, id: int, actor: str, reason: str) -> DocumentSecurityState: ...

    def get_latest_for_source(self, source: str) -> DocumentSecurityState | None: ...

    def list_pending_approval(self, limit: int) -> list[DocumentSecurityState]: ...


class PostgresDocumentSecurityRepository:
    """:class:`DocumentSecurityRepository` backed by Postgres, using the
    app database pool (``app.api.deps.get_db_pool``) - this is app
    metadata, same reasoning as ``rag_ops_config``, not analytics data."""

    def __init__(self, pool: PostgresConnectionPool) -> None:
        self._pool = pool

    def create_pending(
        self, *, source: str, uploaded_by: str, chunk_count: int
    ) -> DocumentSecurityState:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO document_security_state (source, uploaded_by, chunk_count)
                VALUES (%s, %s, %s)
                RETURNING {_COLUMNS}
                """,
                (source, uploaded_by, chunk_count),
            )
            row = cur.fetchone()
            conn.commit()
        return _row_to_state(row)

    def record_scan_result(
        self, *, id: int, status: str, scan_decision: dict[str, Any]
    ) -> DocumentSecurityState:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE document_security_state
                SET status = %s, scan_decision = %s, scanned_at = now()
                WHERE id = %s
                RETURNING {_COLUMNS}
                """,
                (status, Json(scan_decision), id),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"document_security_state row {id} not found")
            conn.commit()
        return _row_to_state(row)

    def approve(self, *, id: int, actor: str) -> DocumentSecurityState:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE document_security_state
                SET status = 'approved', approved_by = %s, approved_at = now()
                WHERE id = %s
                RETURNING {_COLUMNS}
                """,
                (actor, id),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"document_security_state row {id} not found")
            conn.commit()
        return _row_to_state(row)

    def mark_active(self, *, id: int) -> DocumentSecurityState:
        """Distinct from ``approve``: an admin approves a scan_passed
        document, then activation flips the Qdrant payload and this row's
        status together (see
        ``PolicyIngestionSecurityService.approve``) - kept as two steps so
        a failure between them is visible (status stuck at "approved",
        never searchable) rather than silently skipped."""
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"UPDATE document_security_state SET status = 'active' WHERE id = %s "
                f"RETURNING {_COLUMNS}",
                (id,),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"document_security_state row {id} not found")
            conn.commit()
        return _row_to_state(row)

    def reject(self, *, id: int, actor: str, reason: str) -> DocumentSecurityState:
        del actor  # not a separate column - rejected_reason carries the context
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE document_security_state
                SET status = 'rejected', rejected_reason = %s
                WHERE id = %s
                RETURNING {_COLUMNS}
                """,
                (reason, id),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"document_security_state row {id} not found")
            conn.commit()
        return _row_to_state(row)

    def get_latest_for_source(self, source: str) -> DocumentSecurityState | None:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM document_security_state "
                "WHERE source = %s ORDER BY uploaded_at DESC LIMIT 1",
                (source,),
            )
            row = cur.fetchone()
        return None if row is None else _row_to_state(row)

    def list_pending_approval(self, limit: int) -> list[DocumentSecurityState]:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM document_security_state "
                "WHERE status = 'scan_passed' ORDER BY uploaded_at ASC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
        return [_row_to_state(row) for row in rows]


def _row_to_state(row: tuple[Any, ...]) -> DocumentSecurityState:
    return DocumentSecurityState(
        id=row[0],
        source=row[1],
        status=row[2],
        uploaded_by=row[3],
        uploaded_at=row[4],
        scan_decision=row[5],
        scanned_at=row[6],
        approved_by=row[7],
        approved_at=row[8],
        rejected_reason=row[9],
        chunk_count=row[10],
    )
