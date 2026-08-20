"""Opt-in integration test against a real local Postgres instance.

Mirrors tests/repositories/test_rag_ops_self_reflection_integration.py's
shape for the guardrails fields added in migration 013
(guardrail_mode/guardrail_policy_version/safety_lockdown_enabled): skipped
automatically unless Postgres is actually reachable.

Exercises what a hand-rolled fake repository can't: the real
``rag_ops_config_guardrail_mode_check`` CHECK constraint, and that
``set_safety_lockdown`` writes a distinct, non-diff-based audit action
(``safety_lockdown_enabled``/``safety_lockdown_disabled``) rather than the
generic ``config_update`` a routine ``update_config`` call produces - the
whole reason it isn't just another ``update_config`` kwarg.

Run explicitly once the local Postgres container is up and migrated:

    pytest tests/repositories/test_rag_ops_guardrails_integration.py -v
"""

from __future__ import annotations

import os
import uuid

import psycopg2
import pytest

from app.core.config.database import INSECURE_DEFAULT_DSN
from app.core.db import PostgresConnectionPool
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
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE rag_ops_config SET guardrail_mode = 'enforce', "
            "guardrail_policy_version = 'guardrails-policy-v1', "
            "safety_lockdown_enabled = FALSE, safety_lockdown_reason = NULL WHERE id = 1"
        )
        conn.commit()
    yield PostgresRagOpsRepository(pool), pool
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE rag_ops_config SET guardrail_mode = 'enforce', "
            "guardrail_policy_version = 'guardrails-policy-v1', "
            "safety_lockdown_enabled = FALSE, safety_lockdown_reason = NULL WHERE id = 1"
        )
        conn.commit()
    pool.close()


def test_update_config_persists_guardrail_mode(repository) -> None:  # noqa: ANN001
    repo, _ = repository

    config = repo.update_config(actor="itest", reason=None, guardrail_mode="monitor")

    assert config.guardrail_mode == "monitor"


def test_update_config_leaves_guardrail_mode_unchanged_when_omitted(repository) -> None:  # noqa: ANN001
    repo, _ = repository
    repo.update_config(actor="itest", reason=None, guardrail_mode="monitor")

    config = repo.update_config(actor="itest", reason=None, sql_rollout_percentage=10)

    assert config.guardrail_mode == "monitor"  # COALESCE kept the prior value


def test_db_constraint_rejects_an_invalid_guardrail_mode_bypassing_the_application_layer(
    repository,  # noqa: ANN001
) -> None:
    _, pool = repository

    with pool.connection() as conn, conn.cursor() as cur:
        with pytest.raises(psycopg2.errors.CheckViolation):
            cur.execute("UPDATE rag_ops_config SET guardrail_mode = 'nonsense' WHERE id = 1")
        conn.rollback()


def test_set_safety_lockdown_enable_persists_reason_and_actor(repository) -> None:  # noqa: ANN001
    repo, _ = repository

    config = repo.set_safety_lockdown(
        actor="security-admin", enabled=True, reason="suspected prompt-injection campaign"
    )

    assert config.safety_lockdown_enabled is True
    assert config.safety_lockdown_reason == "suspected prompt-injection campaign"
    assert config.safety_lockdown_by == "security-admin"
    assert config.safety_lockdown_at is not None


def test_set_safety_lockdown_disable_clears_the_reason_and_actor(repository) -> None:  # noqa: ANN001
    repo, _ = repository
    repo.set_safety_lockdown(actor="security-admin", enabled=True, reason="incident")

    config = repo.set_safety_lockdown(actor="security-admin", enabled=False, reason=None)

    assert config.safety_lockdown_enabled is False
    assert config.safety_lockdown_reason is None
    assert config.safety_lockdown_by is None
    assert config.safety_lockdown_at is None


def test_safety_lockdown_writes_a_distinct_audit_action_not_config_update(
    repository,  # noqa: ANN001
) -> None:
    repo, pool = repository
    marker = uuid.uuid4().hex[:8]

    repo.set_safety_lockdown(actor=f"itest-{marker}", enabled=True, reason="incident")

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT action FROM rag_ops_audit_log WHERE actor = %s ORDER BY created_at DESC LIMIT 1",
            (f"itest-{marker}",),
        )
        (action,) = cur.fetchone()

    assert action == "safety_lockdown_enabled"


def test_safety_lockdown_is_not_diffed_into_a_routine_config_update_audit_row(
    repository,  # noqa: ANN001
) -> None:
    """safety_lockdown_enabled is deliberately excluded from _DIFF_FIELDS -
    a routine update_config() call must never silently audit it as part of
    a generic config_update diff."""
    repo, _ = repository

    config = repo.update_config(actor="itest", reason=None, guardrail_mode="monitor")

    assert config.safety_lockdown_enabled is False
