"""Opt-in integration test against a real local Postgres instance.

Mirrors tests/repositories/test_rag_ops_guardrails_integration.py's shape:
skipped automatically unless Postgres is actually reachable.

Run explicitly once the local Postgres container is up and migrated:

    pytest tests/repositories/test_security_events_repository.py -v
"""

from __future__ import annotations

import os

import psycopg2
import pytest

from app.core.config.database import INSECURE_DEFAULT_DSN
from app.core.db import PostgresConnectionPool
from app.repositories.security_events_repository import PostgresSecurityEventsRepository

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
    yield PostgresSecurityEventsRepository(pool)
    pool.close()


def test_record_persists_a_sanitized_finding_summary(repository) -> None:  # noqa: ANN001
    event = repository.record(
        actor="alice",
        action="input_block",
        stage="input",
        category="prompt_injection",
        mode="enforce",
        changes={"findings": [{"category": "prompt_injection", "score": 0.98, "detector": "x"}]},
        reason=None,
    )

    assert event.actor == "alice"
    assert event.action == "input_block"
    assert event.stage == "input"
    assert event.category == "prompt_injection"
    assert event.mode == "enforce"
    assert event.changes["findings"][0]["category"] == "prompt_injection"
    assert event.id > 0
    assert event.created_at is not None


def test_record_accepts_a_null_actor() -> None:
    """Context/output guard events have no principal in scope (see
    app.guardrails.context_pipeline/output_pipeline) - actor must be
    nullable, not required."""
    pool = PostgresConnectionPool(_DATABASE_URL)
    repository = PostgresSecurityEventsRepository(pool)

    event = repository.record(
        actor=None,
        action="output_block",
        stage="output",
        category="prompt_leak",
        mode="enforce",
        changes={},
    )

    assert event.actor is None
    pool.close()


def test_list_recent_returns_newest_first(repository) -> None:  # noqa: ANN001
    first = repository.record(
        actor="a", action="input_block", stage="input", category=None, mode="enforce", changes={}
    )
    second = repository.record(
        actor="b", action="output_block", stage="output", category=None, mode="enforce", changes={}
    )

    results = repository.list_recent(limit=2)

    assert results[0].id == second.id
    assert results[1].id == first.id
