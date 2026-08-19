"""SQL proposal persistence: the race-safe approval state machine.

Same connection-pool idiom as ``app.repositories.chat_history_repository``/
``app.repositories.rag_ops_repository`` (``with pool.connection() as conn,
conn.cursor() as cur:``), but against a **row per proposal**, not a
singleton config row - and every state transition in
:meth:`PostgresSQLProposalRepository.lock_for_execution` runs under
``SELECT ... FOR UPDATE`` so two concurrent approval attempts on the same
proposal can't both win (see ``app.core.exceptions.SQLProposalStateError``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from psycopg2.extras import Json

from app.core.db import PostgresConnectionPool
from app.core.exceptions import SQLProposalNotFoundError, SQLProposalStateError
from app.sql.models import SQLProposal, SQLProposalStatus

_PROPOSAL_COLUMNS = (
    "id, username, conversation_id, question, sql_text, sql_fingerprint, "
    "referenced_tables, assumptions, catalog_version, policy_version, "
    "status, expires_at, created_at"
)


class SQLProposalRepository(Protocol):
    def create(self, proposal: SQLProposal) -> SQLProposal: ...

    def get_owned(self, proposal_id: UUID, username: str) -> SQLProposal | None: ...

    def lock_for_execution(
        self, proposal_id: UUID, username: str, now: datetime
    ) -> SQLProposal: ...

    def mark_executed(self, proposal_id: UUID, *, row_count: int, execution_ms: float) -> None: ...

    def mark_failed(self, proposal_id: UUID, *, error_code: str) -> None: ...

    def reject(self, proposal_id: UUID, username: str) -> None: ...


class PostgresSQLProposalRepository:
    def __init__(self, pool: PostgresConnectionPool) -> None:
        self._pool = pool

    def create(self, proposal: SQLProposal) -> SQLProposal:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO sql_query_proposals "
                    "(id, username, conversation_id, question, sql_text, sql_fingerprint, "
                    "referenced_tables, assumptions, catalog_version, policy_version, "
                    "status, expires_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        str(proposal.id),
                        proposal.username,
                        proposal.conversation_id,
                        proposal.question,
                        proposal.sql,
                        proposal.sql_fingerprint,
                        Json(list(proposal.referenced_tables)),
                        Json(list(proposal.assumptions)),
                        proposal.catalog_version,
                        proposal.policy_version,
                        proposal.status.value,
                        proposal.expires_at,
                    ),
                )
            conn.commit()
        return proposal

    def get_owned(self, proposal_id: UUID, username: str) -> SQLProposal | None:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {_PROPOSAL_COLUMNS} FROM sql_query_proposals "
                "WHERE id = %s AND username = %s",
                (str(proposal_id), username),
            )
            row = cur.fetchone()
        return _row_to_proposal(row) if row is not None else None

    def lock_for_execution(self, proposal_id: UUID, username: str, now: datetime) -> SQLProposal:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {_PROPOSAL_COLUMNS} FROM sql_query_proposals "
                    "WHERE id = %s AND username = %s FOR UPDATE",
                    (str(proposal_id), username),
                )
                row = cur.fetchone()
                if row is None:
                    raise SQLProposalNotFoundError(proposal_id)
                proposal = _row_to_proposal(row)

                if proposal.status is not SQLProposalStatus.PROPOSED:
                    raise SQLProposalStateError("proposal_already_consumed")
                # now is timezone-aware (UTC) - callers pass datetime.now(UTC);
                # proposal.expires_at is TIMESTAMPTZ, so this comparison is
                # safe across the pool's session timezone setting.
                if proposal.expires_at <= now:
                    cur.execute(
                        "UPDATE sql_query_proposals SET status = %s WHERE id = %s",
                        (SQLProposalStatus.EXPIRED.value, str(proposal_id)),
                    )
                    conn.commit()
                    raise SQLProposalStateError("proposal_expired")

                cur.execute(
                    "UPDATE sql_query_proposals SET status = %s, approved_at = now() WHERE id = %s",
                    (SQLProposalStatus.EXECUTING.value, str(proposal_id)),
                )
            conn.commit()
        return SQLProposal(
            id=proposal.id,
            username=proposal.username,
            conversation_id=proposal.conversation_id,
            question=proposal.question,
            sql=proposal.sql,
            sql_fingerprint=proposal.sql_fingerprint,
            referenced_tables=proposal.referenced_tables,
            assumptions=proposal.assumptions,
            catalog_version=proposal.catalog_version,
            policy_version=proposal.policy_version,
            status=SQLProposalStatus.EXECUTING,
            expires_at=proposal.expires_at,
            created_at=proposal.created_at,
        )

    def mark_executed(self, proposal_id: UUID, *, row_count: int, execution_ms: float) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sql_query_proposals SET status = %s, executed_at = now(), "
                    "row_count = %s, execution_ms = %s WHERE id = %s",
                    (SQLProposalStatus.EXECUTED.value, row_count, execution_ms, str(proposal_id)),
                )
            conn.commit()

    def mark_failed(self, proposal_id: UUID, *, error_code: str) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sql_query_proposals SET status = %s, error_code = %s WHERE id = %s",
                    (SQLProposalStatus.FAILED.value, error_code, str(proposal_id)),
                )
            conn.commit()

    def reject(self, proposal_id: UUID, username: str) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sql_query_proposals SET status = %s "
                    "WHERE id = %s AND username = %s AND status = %s",
                    (
                        SQLProposalStatus.REJECTED.value,
                        str(proposal_id),
                        username,
                        SQLProposalStatus.PROPOSED.value,
                    ),
                )
                updated = cur.rowcount
            conn.commit()
        if updated == 0:
            raise SQLProposalStateError("proposal_not_pending")


def _row_to_proposal(row: tuple[Any, ...]) -> SQLProposal:
    return SQLProposal(
        id=row[0] if isinstance(row[0], UUID) else UUID(str(row[0])),
        username=row[1],
        conversation_id=row[2],
        question=row[3],
        sql=row[4],
        sql_fingerprint=row[5],
        referenced_tables=tuple(row[6] or []),
        assumptions=tuple(row[7] or []),
        catalog_version=row[8],
        policy_version=row[9],
        status=SQLProposalStatus(row[10]),
        expires_at=row[11],
        created_at=row[12],
    )
