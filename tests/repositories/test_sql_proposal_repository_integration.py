"""Opt-in integration test against a real local Postgres instance.

Mirrors tests/repositories/test_rag_ops_crag_integration.py's shape: skipped
automatically unless Postgres is actually reachable at DATABASE_URL (the
app's own database - proposals are app metadata and live there, not in the
separate analytics database SQL queries execute against; see
app.core.config.sql_settings).

Exercises what a hand-rolled fake repository can't: the actual
SELECT ... FOR UPDATE serialization in
PostgresSQLProposalRepository.lock_for_execution and the
sql_query_proposals status CHECK constraint added in migration 011.

Run explicitly once the local Postgres container (see docker-compose.yml)
is up and migrated:

    pytest tests/repositories/test_sql_proposal_repository_integration.py -v
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta

import psycopg2
import pytest

from app.core.config.database import INSECURE_DEFAULT_DSN
from app.core.db import PostgresConnectionPool
from app.core.exceptions import SQLProposalNotFoundError, SQLProposalStateError
from app.repositories.sql_proposal_repository import PostgresSQLProposalRepository
from app.sql.models import SQLProposal, SQLProposalStatus

_DATABASE_URL = os.environ.get("DATABASE_URL", INSECURE_DEFAULT_DSN)


def _postgres_reachable() -> bool:
    try:
        conn = psycopg2.connect(_DATABASE_URL, connect_timeout=2)
        conn.close()
        return True
    except Exception:  # noqa: BLE001 - any connection failure means "skip"
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_reachable(),
    reason=f"Postgres not reachable at {_DATABASE_URL} - skipping integration test",
)


@pytest.fixture
def repository():
    pool = PostgresConnectionPool(_DATABASE_URL)
    username = f"itest-sql-{uuid.uuid4().hex[:8]}"
    with pool.connection() as conn, conn.cursor() as cur:
        # conversations.username FKs to users.username - a real (if
        # throwaway) user row is required, same as any other repository
        # integration test that touches conversations.
        cur.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (%s, 'x', TRUE)",
            (username,),
        )
        conn.commit()
    yield PostgresSQLProposalRepository(pool), pool, username
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM sql_query_proposals WHERE conversation_id IN "
            "(SELECT id FROM conversations WHERE username = %s)",
            (username,),
        )
        cur.execute("DELETE FROM conversations WHERE username = %s", (username,))
        cur.execute("DELETE FROM users WHERE username = %s", (username,))
        conn.commit()
    pool.close()


def _make_conversation(pool: PostgresConnectionPool, username: str) -> int:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO conversations (username, title) VALUES (%s, %s) RETURNING id",
            (username, "itest conversation"),
        )
        (conversation_id,) = cur.fetchone()
        conn.commit()
    return conversation_id


def _proposal(conversation_id: int, *, username: str, expires_in_seconds: int = 300) -> SQLProposal:
    now = datetime.now(UTC)
    return SQLProposal(
        id=uuid.uuid4(),
        username=username,
        conversation_id=conversation_id,
        question="how many accounts opened last month?",
        sql="SELECT a.id FROM approved_analytics.accounts a LIMIT 10",
        sql_fingerprint="f" * 64,
        referenced_tables=("approved_analytics.accounts",),
        assumptions=(),
        catalog_version=1,
        policy_version="sql-policy-v1",
        status=SQLProposalStatus.PROPOSED,
        expires_at=now + timedelta(seconds=expires_in_seconds),
        created_at=now,
    )


def test_create_then_get_owned_round_trips(repository) -> None:  # noqa: ANN001
    repo, pool, username = repository
    conversation_id = _make_conversation(pool, username)
    proposal = _proposal(conversation_id, username=username)

    repo.create(proposal)
    fetched = repo.get_owned(proposal.id, username)

    assert fetched is not None
    assert fetched.sql == proposal.sql
    assert fetched.status is SQLProposalStatus.PROPOSED


def test_get_owned_returns_none_for_a_different_username(repository) -> None:  # noqa: ANN001
    repo, pool, username = repository
    conversation_id = _make_conversation(pool, username)
    proposal = _proposal(conversation_id, username=username)
    repo.create(proposal)

    assert repo.get_owned(proposal.id, "someone-else") is None


def test_lock_for_execution_transitions_to_executing(repository) -> None:  # noqa: ANN001
    repo, pool, username = repository
    conversation_id = _make_conversation(pool, username)
    proposal = _proposal(conversation_id, username=username)
    repo.create(proposal)

    locked = repo.lock_for_execution(proposal.id, username, datetime.now(UTC))

    assert locked.status is SQLProposalStatus.EXECUTING


def test_lock_for_execution_twice_raises_on_the_second_call(repository) -> None:  # noqa: ANN001
    """The race-safety guarantee this whole repository exists for: a second
    concurrent approval attempt must not also succeed."""
    repo, pool, username = repository
    conversation_id = _make_conversation(pool, username)
    proposal = _proposal(conversation_id, username=username)
    repo.create(proposal)
    repo.lock_for_execution(proposal.id, username, datetime.now(UTC))

    with pytest.raises(SQLProposalStateError, match="proposal_already_consumed"):
        repo.lock_for_execution(proposal.id, username, datetime.now(UTC))


def test_lock_for_execution_on_unknown_id_raises_not_found(repository) -> None:  # noqa: ANN001
    repo, _, username = repository

    with pytest.raises(SQLProposalNotFoundError):
        repo.lock_for_execution(uuid.uuid4(), username, datetime.now(UTC))


def test_lock_for_execution_on_expired_proposal_marks_it_expired_and_raises(repository) -> None:  # noqa: ANN001
    repo, pool, username = repository
    conversation_id = _make_conversation(pool, username)
    proposal = _proposal(conversation_id, username=username, expires_in_seconds=-1)
    repo.create(proposal)

    with pytest.raises(SQLProposalStateError, match="proposal_expired"):
        repo.lock_for_execution(proposal.id, username, datetime.now(UTC))

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT status FROM sql_query_proposals WHERE id = %s", (str(proposal.id),))
        (status,) = cur.fetchone()
    assert status == SQLProposalStatus.EXPIRED.value


def test_mark_executed_records_row_count_and_execution_ms(repository) -> None:  # noqa: ANN001
    repo, pool, username = repository
    conversation_id = _make_conversation(pool, username)
    proposal = _proposal(conversation_id, username=username)
    repo.create(proposal)
    repo.lock_for_execution(proposal.id, username, datetime.now(UTC))

    repo.mark_executed(proposal.id, row_count=42, execution_ms=12.5)

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT status, row_count, execution_ms FROM sql_query_proposals WHERE id = %s",
            (str(proposal.id),),
        )
        status, row_count, execution_ms = cur.fetchone()
    assert status == SQLProposalStatus.EXECUTED.value
    assert row_count == 42
    assert execution_ms == 12.5


def test_reject_only_succeeds_from_proposed_state(repository) -> None:  # noqa: ANN001
    repo, pool, username = repository
    conversation_id = _make_conversation(pool, username)
    proposal = _proposal(conversation_id, username=username)
    repo.create(proposal)
    repo.lock_for_execution(proposal.id, username, datetime.now(UTC))

    with pytest.raises(SQLProposalStateError, match="proposal_not_pending"):
        repo.reject(proposal.id, username)


def test_db_constraint_rejects_an_unknown_status_bypassing_the_application_layer(
    repository,  # noqa: ANN001
) -> None:
    _, pool, username = repository
    conversation_id = _make_conversation(pool, username)
    proposal = _proposal(conversation_id, username=username)
    with pool.connection() as conn, conn.cursor() as cur:
        with pytest.raises(psycopg2.errors.CheckViolation):
            cur.execute(
                "INSERT INTO sql_query_proposals "
                "(id, username, conversation_id, question, sql_text, sql_fingerprint, "
                "catalog_version, policy_version, status, expires_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    str(proposal.id),
                    proposal.username,
                    conversation_id,
                    proposal.question,
                    proposal.sql,
                    proposal.sql_fingerprint,
                    proposal.catalog_version,
                    proposal.policy_version,
                    "not_a_real_status",
                    proposal.expires_at,
                ),
            )
        conn.rollback()
