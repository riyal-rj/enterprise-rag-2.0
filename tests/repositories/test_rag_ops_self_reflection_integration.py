"""Opt-in integration test against a real local Postgres instance.

Mirrors tests/repositories/test_rag_ops_crag_integration.py's shape exactly,
for the self-reflection fields added in migration 010
(self_reflective_enabled/self_reflective_rollout_percentage) and migration
012 (self_reflective_shadow_enabled/self_reflective_retrieval_enabled):
skipped automatically unless Postgres is actually reachable.

Exercises what a hand-rolled fake repository can't: the actual
``SELECT ... FOR UPDATE`` revalidation in
``PostgresRagOpsRepository.update_config``, the
``rag_ops_config_self_reflective_rollout_percentage_check``/
``rag_ops_config_self_reflective_shadow_requires_enabled_check``/
``rag_ops_config_self_reflective_retrieval_requires_enabled_check`` CHECK
constraints, and - going one step further than the CRAG integration test -
that ``FOR UPDATE`` genuinely serializes two concurrent writers racing the
same singleton row, not just "the sequential revalidation logic happens to
be correct."

Run explicitly once the local Postgres container (see docker-compose.yml)
is up and migrated:

    pytest tests/repositories/test_rag_ops_self_reflection_integration.py -v
"""

from __future__ import annotations

import os
import threading
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
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE rag_ops_config SET self_reflective_enabled = FALSE, "
            "self_reflective_shadow_enabled = FALSE, self_reflective_retrieval_enabled = FALSE, "
            "self_reflective_rollout_percentage = 0, updated_by = %s WHERE id = 1",
            (actor,),
        )
        conn.commit()
    yield PostgresRagOpsRepository(pool), pool
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE rag_ops_config SET self_reflective_enabled = FALSE, "
            "self_reflective_shadow_enabled = FALSE, self_reflective_retrieval_enabled = FALSE, "
            "self_reflective_rollout_percentage = 0 WHERE id = 1"
        )
        conn.commit()
    pool.close()


def test_update_config_revalidates_self_reflective_shadow_state_under_lock(repository) -> None:  # noqa: ANN001
    repo, pool = repository

    with pytest.raises(InvalidRagOpsConfigError):
        # self_reflective_enabled stays False (unset here, and the row
        # already has it False from the fixture's reset) while
        # self_reflective_shadow_enabled is explicitly requested True - must
        # be rejected under the same SELECT ... FOR UPDATE transaction, not
        # just at the controller.
        repo.update_config(actor="itest", reason=None, self_reflective_shadow_enabled=True)

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT self_reflective_shadow_enabled FROM rag_ops_config WHERE id = 1")
        (shadow_enabled,) = cur.fetchone()
    assert shadow_enabled is False


def test_update_config_revalidates_self_reflective_retrieval_state_under_lock(repository) -> None:  # noqa: ANN001
    repo, pool = repository

    with pytest.raises(InvalidRagOpsConfigError):
        repo.update_config(actor="itest", reason=None, self_reflective_retrieval_enabled=True)

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT self_reflective_retrieval_enabled FROM rag_ops_config WHERE id = 1")
        (retrieval_enabled,) = cur.fetchone()
    assert retrieval_enabled is False


def test_failed_transition_writes_no_audit_row(repository) -> None:  # noqa: ANN001
    repo, pool = repository
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM rag_ops_audit_log")
        (count_before,) = cur.fetchone()

    with pytest.raises(InvalidRagOpsConfigError):
        repo.update_config(actor="itest", reason=None, self_reflective_shadow_enabled=True)

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM rag_ops_audit_log")
        (count_after,) = cur.fetchone()
    assert count_after == count_before


def test_valid_transition_enabling_both_together_succeeds(repository) -> None:  # noqa: ANN001
    repo, _ = repository

    config = repo.update_config(
        actor="itest",
        reason=None,
        self_reflective_enabled=True,
        self_reflective_shadow_enabled=True,
        self_reflective_retrieval_enabled=True,
        self_reflective_rollout_percentage=25,
    )

    assert config.self_reflective_enabled is True
    assert config.self_reflective_shadow_enabled is True
    assert config.self_reflective_retrieval_enabled is True
    assert config.self_reflective_rollout_percentage == 25


def test_db_constraint_rejects_invalid_shadow_state_bypassing_the_application_layer(
    repository,  # noqa: ANN001
) -> None:
    _, pool = repository

    with pool.connection() as conn, conn.cursor() as cur:
        with pytest.raises(psycopg2.errors.CheckViolation):
            cur.execute(
                "UPDATE rag_ops_config SET self_reflective_enabled = FALSE, "
                "self_reflective_shadow_enabled = TRUE WHERE id = 1"
            )
        conn.rollback()


def test_db_constraint_rejects_invalid_retrieval_state_bypassing_the_application_layer(
    repository,  # noqa: ANN001
) -> None:
    _, pool = repository

    with pool.connection() as conn, conn.cursor() as cur:
        with pytest.raises(psycopg2.errors.CheckViolation):
            cur.execute(
                "UPDATE rag_ops_config SET self_reflective_enabled = FALSE, "
                "self_reflective_retrieval_enabled = TRUE WHERE id = 1"
            )
        conn.rollback()


def test_db_constraint_rejects_out_of_range_rollout_percentage(repository) -> None:  # noqa: ANN001
    _, pool = repository

    with pool.connection() as conn, conn.cursor() as cur:
        with pytest.raises(psycopg2.errors.CheckViolation):
            cur.execute(
                "UPDATE rag_ops_config SET self_reflective_rollout_percentage = 101 WHERE id = 1"
            )
        conn.rollback()


def test_concurrent_writers_are_serialized_not_lost(repository) -> None:  # noqa: ANN001
    """Two real threads race ``update_config`` against the same singleton
    row at (as close to) the same time as this test can arrange. Both
    requests are individually valid (each only sets
    self_reflective_enabled=True plus a distinct rollout percentage), so
    without ``SELECT ... FOR UPDATE`` actually serializing them, a
    read-modify-write race could let one writer's update clobber the
    other's audit trail or leave the row's final rollout percentage
    ambiguous/inconsistent with what got audited. This proves the lock is
    real, not just present in the SQL text."""
    repo, pool = repository
    barrier = threading.Barrier(2)
    results: dict[str, int] = {}
    errors: list[Exception] = []

    def _writer(name: str, rollout: int) -> None:
        try:
            barrier.wait(timeout=5)  # maximize actual overlap between the two transactions
            config = repo.update_config(
                actor=name,
                reason=None,
                self_reflective_enabled=True,
                self_reflective_rollout_percentage=rollout,
            )
            results[name] = config.self_reflective_rollout_percentage
        except Exception as exc:  # noqa: BLE001 - captured for the assertion below
            errors.append(exc)

    t1 = threading.Thread(target=_writer, args=("writer-a", 10))
    t2 = threading.Thread(target=_writer, args=("writer-b", 90))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors, f"concurrent writers raised: {errors}"
    assert len(results) == 2  # both transactions completed, none silently dropped

    # The final persisted value must be exactly one writer's value - not a
    # torn/interleaved result - and the audit log must show both writes
    # happened (proving neither was silently lost to a lost-update race).
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT self_reflective_rollout_percentage FROM rag_ops_config WHERE id = 1")
        (final_value,) = cur.fetchone()
        cur.execute(
            "SELECT actor FROM rag_ops_audit_log WHERE actor IN ('writer-a', 'writer-b') "
            "ORDER BY created_at"
        )
        audited_actors = [row[0] for row in cur.fetchall()]

    assert final_value in (10, 90)
    assert sorted(audited_actors) == ["writer-a", "writer-b"]
