"""Opt-in integration test against a real local Postgres instance.

Mirrors tests/repositories/test_vector_repository_integration.py's shape:
skipped automatically unless Postgres is actually reachable at
``DATABASE_URL`` (default matches ``DatabaseSettings.INSECURE_DEFAULT_DSN``)
- no separate opt-in flag needed, and the standard ``pytest tests/ -q`` run
doesn't newly depend on a running Postgres container.

Exercises what a hand-rolled fake repository can't: the actual
``SELECT ... FOR UPDATE`` revalidation in
``PostgresRagOpsRepository.update_config`` (see app.repositories.rag_ops_repository)
and the ``rag_ops_config_crag_web_requires_crag_check`` CHECK constraint
added in migration 008 - the final backstop against any writer, including
one that bypasses the application layer entirely.

Run explicitly once the local Postgres container (see docker-compose.yml)
is up and migrated:

    pytest tests/repositories/test_rag_ops_crag_integration.py -v
"""

from __future__ import annotations

import os
import uuid

import psycopg2
import pytest

from app.core.config.database import INSECURE_DEFAULT_DSN
from app.core.db import PostgresConnectionPool
from app.core.exceptions import InvalidRagOpsConfigError
from app.repositories.rag_ops_repository import PostgresRagOpsRepository

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
    actor = f"itest-{uuid.uuid4().hex[:8]}"
    # Reset the singleton row to a known, CRAG-disabled baseline before each
    # test, so this file doesn't depend on run order or another suite's
    # leftover state.
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE rag_ops_config SET crag_enabled = FALSE, crag_web_enabled = FALSE, "
            "crag_rollout_percentage = 0, updated_by = %s WHERE id = 1",
            (actor,),
        )
        conn.commit()
    yield PostgresRagOpsRepository(pool), pool
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE rag_ops_config SET crag_enabled = FALSE, crag_web_enabled = FALSE, "
            "crag_rollout_percentage = 0 WHERE id = 1"
        )
        conn.commit()
    pool.close()


def test_update_config_revalidates_crag_state_under_lock(repository) -> None:  # noqa: ANN001
    repo, pool = repository

    with pytest.raises(InvalidRagOpsConfigError):
        # crag_enabled stays False (unset in this call, and the row already
        # has it False from the fixture's reset) while crag_web_enabled is
        # explicitly requested True - must be rejected under the same
        # SELECT ... FOR UPDATE transaction, not just at the controller.
        repo.update_config(actor="itest", reason=None, crag_web_enabled=True)

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT crag_web_enabled FROM rag_ops_config WHERE id = 1")
        (crag_web_enabled,) = cur.fetchone()
    assert crag_web_enabled is False


def test_failed_transition_writes_no_audit_row(repository) -> None:  # noqa: ANN001
    repo, pool = repository
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM rag_ops_audit_log")
        (count_before,) = cur.fetchone()

    with pytest.raises(InvalidRagOpsConfigError):
        repo.update_config(actor="itest", reason=None, crag_web_enabled=True)

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM rag_ops_audit_log")
        (count_after,) = cur.fetchone()
    assert count_after == count_before


def test_valid_transition_enabling_both_together_succeeds(repository) -> None:  # noqa: ANN001
    repo, _ = repository

    config = repo.update_config(
        actor="itest", reason=None, crag_enabled=True, crag_web_enabled=True
    )

    assert config.crag_enabled is True
    assert config.crag_web_enabled is True


def test_db_constraint_rejects_invalid_state_bypassing_the_application_layer(repository) -> None:  # noqa: ANN001
    _, pool = repository

    with pool.connection() as conn, conn.cursor() as cur:
        with pytest.raises(psycopg2.errors.CheckViolation):
            cur.execute(
                "UPDATE rag_ops_config SET crag_enabled = FALSE, crag_web_enabled = TRUE WHERE id = 1"
            )
        conn.rollback()
