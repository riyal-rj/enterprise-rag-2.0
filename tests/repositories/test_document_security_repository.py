"""Opt-in integration test against a real local Postgres instance.

Mirrors tests/repositories/test_rag_ops_guardrails_integration.py's shape:
skipped automatically unless Postgres is actually reachable. Exercises the
per-document ingestion-quarantine lifecycle
(pending_scan -> scan_passed/scan_failed -> approved -> active, or
rejected) - see app.services.policy_ingestion_security_service and
app/seed/migrations/014_create_document_security_state.sql.

Run explicitly once the local Postgres container is up and migrated:

    pytest tests/repositories/test_document_security_repository.py -v
"""

from __future__ import annotations

import os
import uuid

import psycopg2
import pytest

from app.core.config.database import INSECURE_DEFAULT_DSN
from app.core.db import PostgresConnectionPool
from app.repositories.document_security_repository import PostgresDocumentSecurityRepository

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
    yield PostgresDocumentSecurityRepository(pool)
    pool.close()


def _source() -> str:
    return f"itest-{uuid.uuid4().hex[:8]}.pdf"


def test_create_pending_starts_in_pending_scan_status(repository) -> None:  # noqa: ANN001
    source = _source()

    state = repository.create_pending(source=source, uploaded_by="admin", chunk_count=5)

    assert state.source == source
    assert state.status == "pending_scan"
    assert state.uploaded_by == "admin"
    assert state.chunk_count == 5
    assert state.scan_decision is None


def test_record_scan_result_persists_the_decision_and_advances_status(repository) -> None:  # noqa: ANN001
    source = _source()
    state = repository.create_pending(source=source, uploaded_by="admin", chunk_count=1)

    updated = repository.record_scan_result(
        id=state.id,
        status="scan_passed",
        scan_decision={"action": "allow", "findings": []},
    )

    assert updated.status == "scan_passed"
    assert updated.scan_decision == {"action": "allow", "findings": []}
    assert updated.scanned_at is not None


def test_full_lifecycle_pending_to_approved_to_active(repository) -> None:  # noqa: ANN001
    source = _source()
    state = repository.create_pending(source=source, uploaded_by="admin", chunk_count=2)
    repository.record_scan_result(id=state.id, status="scan_passed", scan_decision={})

    approved = repository.approve(id=state.id, actor="security-admin")
    assert approved.status == "approved"
    assert approved.approved_by == "security-admin"
    assert approved.approved_at is not None

    active = repository.mark_active(id=state.id)
    assert active.status == "active"


def test_reject_records_the_reason(repository) -> None:  # noqa: ANN001
    source = _source()
    state = repository.create_pending(source=source, uploaded_by="admin", chunk_count=1)

    rejected = repository.reject(id=state.id, actor="admin", reason="prompt injection detected")

    assert rejected.status == "rejected"
    assert rejected.rejected_reason == "prompt injection detected"


def test_get_latest_for_source_returns_the_most_recent_upload(repository) -> None:  # noqa: ANN001
    source = _source()
    first = repository.create_pending(source=source, uploaded_by="admin", chunk_count=1)
    repository.reject(id=first.id, actor="admin", reason="bad content")
    second = repository.create_pending(source=source, uploaded_by="admin", chunk_count=3)

    latest = repository.get_latest_for_source(source)

    assert latest is not None
    assert latest.id == second.id
    assert latest.status == "pending_scan"


def test_get_latest_for_source_returns_none_for_an_unknown_source(repository) -> None:  # noqa: ANN001
    assert repository.get_latest_for_source(_source()) is None


def test_reupload_creates_a_new_row_not_an_update(repository) -> None:  # noqa: ANN001
    """A re-upload of the same filename must preserve the audit history of
    every prior version - see the migration's own docstring."""
    source = _source()
    first = repository.create_pending(source=source, uploaded_by="admin", chunk_count=1)
    second = repository.create_pending(source=source, uploaded_by="admin", chunk_count=2)

    assert first.id != second.id


def test_list_pending_approval_returns_only_scan_passed_documents(repository) -> None:  # noqa: ANN001
    passed_source = _source()
    failed_source = _source()
    pending = repository.create_pending(source=passed_source, uploaded_by="admin", chunk_count=1)
    repository.record_scan_result(id=pending.id, status="scan_passed", scan_decision={})
    failed = repository.create_pending(source=failed_source, uploaded_by="admin", chunk_count=1)
    repository.record_scan_result(id=failed.id, status="scan_failed", scan_decision={})

    results = repository.list_pending_approval(limit=100)

    result_sources = {r.source for r in results}
    assert passed_source in result_sources
    assert failed_source not in result_sources
