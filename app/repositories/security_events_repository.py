"""Persistence for the guardrails security-event audit trail - see
``app.guardrails`` and ``app/seed/migrations/015_create_security_events.sql``.

Copies ``app.repositories.rag_ops_repository``'s audit-insert pattern
(``_insert_audit``) almost exactly, but as its own table/repository: these
rows are written from request-handling code (every BLOCK/REDACT guardrail
decision), not from admin config mutations, and at much higher volume.

Deliberately typed with plain ``str`` params (not the ``GuardrailStage``/
``GuardrailCategory`` enums from ``app.guardrails.contracts``) so this
module has no dependency on ``app.guardrails`` - callers pass
``.value``/``None``, the same way ``RagOpsAuditEntry.action`` is a plain
string, not an enum.
"""

from __future__ import annotations

from typing import Any, Protocol

from psycopg2.extras import Json

from app.core.db import PostgresConnectionPool
from app.models.security_event import SecurityEvent


class SecurityEventsRepository(Protocol):
    def record(
        self,
        *,
        actor: str | None,
        action: str,
        stage: str,
        category: str | None,
        mode: str,
        changes: dict[str, Any],
        reason: str | None = None,
    ) -> SecurityEvent: ...

    def list_recent(self, limit: int) -> list[SecurityEvent]: ...


class PostgresSecurityEventsRepository:
    def __init__(self, pool: PostgresConnectionPool) -> None:
        self._pool = pool

    def record(
        self,
        *,
        actor: str | None,
        action: str,
        stage: str,
        category: str | None,
        mode: str,
        changes: dict[str, Any],
        reason: str | None = None,
    ) -> SecurityEvent:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO security_events
                    (actor, action, stage, category, mode, changes, reason)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id, actor, action, stage, category, mode, changes, reason, created_at
                """,
                (actor, action, stage, category, mode, Json(changes), reason),
            )
            row = cur.fetchone()
            conn.commit()
        return _row_to_event(row)

    def list_recent(self, limit: int) -> list[SecurityEvent]:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, actor, action, stage, category, mode, changes, reason, created_at "
                "FROM security_events ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
        return [_row_to_event(row) for row in rows]


def _row_to_event(row: tuple[Any, ...]) -> SecurityEvent:
    return SecurityEvent(
        id=row[0],
        actor=row[1],
        action=row[2],
        stage=row[3],
        category=row[4],
        mode=row[5],
        changes=row[6] or {},
        reason=row[7],
        created_at=row[8],
    )
