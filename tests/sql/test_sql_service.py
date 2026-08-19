"""Tests for SQLService's propose()/approve_and_execute() orchestration,
against hand-rolled fakes for every collaborator - same convention as the
rest of this codebase's service-level tests (see e.g.
tests/rag_services/test_crag_refiner.py's local _FakeLLMClient)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID, uuid4

import psycopg2.errors
import pytest

from app.core.exceptions import SQLGenerationFailedError, SQLProposalStateError
from app.sql.catalog import StaticSQLCatalog
from app.sql.models import (
    CatalogSnapshot,
    ExplainAssessment,
    SanitizedSQLResult,
    SQLExecutionResult,
    SQLGeneration,
    SQLPrincipal,
    SQLProposal,
    SQLProposalStatus,
    ValidatedSQL,
)
from app.sql.sql_generator import VannaSQLGenerator
from app.sql.sql_policy import SQLPolicyViolation
from app.sql.sql_service import SQLService


def _catalog(version: int = 1) -> CatalogSnapshot:
    return CatalogSnapshot(version=version, columns=(), relationships=(), business_rules=())


def _principal() -> SQLPrincipal:
    return SQLPrincipal(username="admin", is_admin=True)


def _validated(
    *, fingerprint: str = "f" * 64, policy_version: str = "sql-policy-v1"
) -> ValidatedSQL:
    return ValidatedSQL(
        sql="SELECT a.id FROM approved_analytics.accounts a LIMIT 10",
        fingerprint=fingerprint,
        referenced_tables=("approved_analytics.accounts",),
        referenced_columns=("approved_analytics.accounts.id",),
        row_limit=10,
        policy_version=policy_version,
    )


class _FakeCatalog:
    def __init__(self, version: int = 1) -> None:
        self.version = version

    def current(self, principal: SQLPrincipal) -> CatalogSnapshot:
        del principal
        return _catalog(self.version)


class _FakeGenerator:
    def __init__(self, sql_sequence: list[str]) -> None:
        self._sql_sequence = list(sql_sequence)
        self.calls: list[dict[str, object]] = []
        self.sync_catalog_calls = 0
        self.sync_examples_calls = 0

    def sync_catalog(self, catalog_service: object, snapshot: CatalogSnapshot) -> None:
        del catalog_service, snapshot
        self.sync_catalog_calls += 1

    def sync_examples(self, examples: list[object]) -> None:
        del examples
        self.sync_examples_calls += 1

    def generate(
        self,
        *,
        question: str,
        principal: SQLPrincipal,
        catalog: CatalogSnapshot,
        rendered_catalog: str,
        examples: list[object],
        correction_code: str | None = None,
    ) -> SQLGeneration:
        del question, principal, catalog, rendered_catalog, examples
        self.calls.append({"correction_code": correction_code})
        sql = self._sql_sequence[len(self.calls) - 1]
        return SQLGeneration(sql=sql, tables=(), assumptions=(), usage_tokens=5, duration_ms=1.0)


class _FakePolicy:
    """Treats the literal string "BAD" as an unpolicy-compliant query, and
    "MISMATCH" as one whose recomputed fingerprint differs from what was
    originally stored - everything else validates to a fixed ValidatedSQL."""

    def validate_and_rewrite(
        self, sql: str, *, catalog: CatalogSnapshot, principal: SQLPrincipal
    ) -> ValidatedSQL:
        del catalog, principal
        if sql == "BAD":
            raise SQLPolicyViolation("select_star_forbidden")
        if sql == "MISMATCH":
            return _validated(fingerprint="mismatched-fingerprint".ljust(64, "0"))
        return _validated()


class _FakeExecutor:
    def __init__(self, *, execute_error: Exception | None = None) -> None:
        self._execute_error = execute_error
        self.executed = False

    def explain(self, query: ValidatedSQL, principal: SQLPrincipal) -> ExplainAssessment:
        del query, principal
        return ExplainAssessment(total_cost=1.0, plan_rows=1, plan_width=8)

    def execute(self, query: ValidatedSQL, principal: SQLPrincipal) -> SQLExecutionResult:
        del query, principal
        if self._execute_error is not None:
            raise self._execute_error
        self.executed = True
        return SQLExecutionResult(
            columns=("id",),
            rows=((1,),),
            row_count=1,
            truncated=False,
            duration_ms=2.0,
            bytes_returned=8,
        )


class _FakeResultPolicy:
    def apply(
        self, result: SQLExecutionResult, principal: SQLPrincipal, catalog: CatalogSnapshot
    ) -> SanitizedSQLResult:
        del principal, catalog
        return SanitizedSQLResult(
            columns=result.columns,
            rows=result.rows,
            row_count=result.row_count,
            truncated=result.truncated,
            snapshot_at=datetime.now(UTC),
        )


class _FakeAnswerer:
    def answer(self, *, question: str, query: ValidatedSQL, result: SanitizedSQLResult) -> str:
        del question, query
        return f"{result.row_count} row(s) found."


class _FakeProposalRepository:
    def __init__(self) -> None:
        self._proposals: dict[UUID, SQLProposal] = {}
        self.mark_failed_calls: list[str] = []
        self.mark_executed_calls: list[dict[str, object]] = []

    def create(self, proposal: SQLProposal) -> SQLProposal:
        self._proposals[proposal.id] = proposal
        return proposal

    def get_owned(self, proposal_id: UUID, username: str) -> SQLProposal | None:
        proposal = self._proposals.get(proposal_id)
        if proposal is None or proposal.username != username:
            return None
        return proposal

    def lock_for_execution(self, proposal_id: UUID, username: str, now: datetime) -> SQLProposal:
        proposal = self.get_owned(proposal_id, username)
        if proposal is None:
            raise KeyError(proposal_id)
        if proposal.status is not SQLProposalStatus.PROPOSED:
            raise SQLProposalStateError("proposal_already_consumed")
        if proposal.expires_at <= now:
            raise SQLProposalStateError("proposal_expired")
        executing = SQLProposal(**{**proposal.__dict__, "status": SQLProposalStatus.EXECUTING})
        self._proposals[proposal_id] = executing
        return executing

    def mark_executed(self, proposal_id: UUID, *, row_count: int, execution_ms: float) -> None:
        self.mark_executed_calls.append({"row_count": row_count, "execution_ms": execution_ms})
        proposal = self._proposals[proposal_id]
        self._proposals[proposal_id] = SQLProposal(
            **{**proposal.__dict__, "status": SQLProposalStatus.EXECUTED}
        )

    def mark_failed(self, proposal_id: UUID, *, error_code: str) -> None:
        self.mark_failed_calls.append(error_code)
        proposal = self._proposals[proposal_id]
        self._proposals[proposal_id] = SQLProposal(
            **{**proposal.__dict__, "status": SQLProposalStatus.FAILED}
        )

    def reject(self, proposal_id: UUID, username: str) -> None:
        proposal = self.get_owned(proposal_id, username)
        if proposal is None:
            raise KeyError(proposal_id)
        self._proposals[proposal_id] = SQLProposal(
            **{**proposal.__dict__, "status": SQLProposalStatus.REJECTED}
        )


class _FakeExampleRepository:
    def list_active(self, limit: int) -> list[object]:
        del limit
        return []


def _service(
    *,
    sql_sequence: list[str] | None = None,
    catalog_version: int = 1,
    execute_error: Exception | None = None,
    max_generation_attempts: int = 2,
) -> tuple[SQLService, _FakeGenerator, _FakeProposalRepository, _FakeExecutor]:
    generator = _FakeGenerator(sql_sequence or ["SELECT a.id FROM approved_analytics.accounts a"])
    proposals = _FakeProposalRepository()
    executor = _FakeExecutor(execute_error=execute_error)
    service = SQLService(
        catalog=cast(StaticSQLCatalog, _FakeCatalog(catalog_version)),
        generator=cast(VannaSQLGenerator, generator),
        policy=cast(object, _FakePolicy()),  # type: ignore[arg-type]
        executor=cast(object, executor),  # type: ignore[arg-type]
        result_policy=cast(object, _FakeResultPolicy()),  # type: ignore[arg-type]
        answerer=cast(object, _FakeAnswerer()),  # type: ignore[arg-type]
        proposals=cast(object, proposals),  # type: ignore[arg-type]
        examples=cast(object, _FakeExampleRepository()),  # type: ignore[arg-type]
        proposal_ttl_seconds=300,
        max_generation_attempts=max_generation_attempts,
        max_examples=10,
    )
    return service, generator, proposals, executor


def test_propose_creates_a_proposal_on_first_successful_generation() -> None:
    service, generator, _, _ = _service()

    proposal = service.propose(
        principal=_principal(), conversation_id=1, question="how many accounts?"
    )

    assert proposal.status is SQLProposalStatus.PROPOSED
    assert proposal.catalog_version == 1
    assert len(generator.calls) == 1
    assert generator.calls[0]["correction_code"] is None
    assert generator.sync_catalog_calls == 1
    assert generator.sync_examples_calls == 1


def test_propose_retries_with_correction_code_after_a_policy_violation() -> None:
    service, generator, _, _ = _service(
        sql_sequence=["BAD", "SELECT a.id FROM approved_analytics.accounts a"]
    )

    proposal = service.propose(principal=_principal(), conversation_id=1, question="q")

    assert proposal.status is SQLProposalStatus.PROPOSED
    assert len(generator.calls) == 2
    assert generator.calls[0]["correction_code"] is None
    assert generator.calls[1]["correction_code"] == "select_star_forbidden"


def test_propose_raises_after_exhausting_generation_attempts() -> None:
    service, generator, _, _ = _service(sql_sequence=["BAD", "BAD"], max_generation_attempts=2)

    with pytest.raises(SQLGenerationFailedError):
        service.propose(principal=_principal(), conversation_id=1, question="q")

    assert len(generator.calls) == 2


def _create_proposal(service: SQLService) -> SQLProposal:
    return service.propose(principal=_principal(), conversation_id=1, question="how many accounts?")


def test_approve_and_execute_happy_path() -> None:
    service, _, proposals, executor = _service()
    proposal = _create_proposal(service)

    completed = service.approve_and_execute(principal=_principal(), proposal_id=proposal.id)

    assert completed.answer == "1 row(s) found."
    assert completed.proposal.status is SQLProposalStatus.EXECUTED
    assert executor.executed is True
    assert len(proposals.mark_executed_calls) == 1


def test_approve_and_execute_rejects_when_catalog_version_changed() -> None:
    service, _, proposals, _ = _service(catalog_version=1)
    proposal = _create_proposal(service)
    # Simulate a catalog refresh landing between proposal and approval by
    # swapping in a catalog that reports a newer version.
    service._catalog = cast(StaticSQLCatalog, _FakeCatalog(version=2))  # noqa: SLF001

    with pytest.raises(SQLProposalStateError, match="catalog_changed"):
        service.approve_and_execute(principal=_principal(), proposal_id=proposal.id)

    assert proposals.mark_failed_calls == ["catalog_changed"]


def test_approve_and_execute_rejects_on_fingerprint_mismatch() -> None:
    """Simulates the proposal row being tampered with (or corrupted)
    between proposal and approval: the stored SQL still re-validates fine,
    but its stored fingerprint no longer matches what re-validation
    recomputes - the exact TOCTOU case the blueprint's approval binding
    exists to catch."""
    service, _, proposals, _ = _service()
    proposal = _create_proposal(service)
    stored = proposals._proposals[proposal.id]  # noqa: SLF001
    proposals._proposals[proposal.id] = SQLProposal(  # noqa: SLF001
        **{**stored.__dict__, "sql_fingerprint": "definitely-not-the-real-one".ljust(64, "0")}
    )

    with pytest.raises(SQLProposalStateError, match="fingerprint_mismatch"):
        service.approve_and_execute(principal=_principal(), proposal_id=proposal.id)

    assert proposals.mark_failed_calls == ["fingerprint_mismatch"]


def test_approve_and_execute_maps_query_canceled_to_statement_timeout() -> None:
    service, _, proposals, _ = _service(
        execute_error=psycopg2.errors.QueryCanceled("canceling statement due to statement timeout")
    )
    proposal = _create_proposal(service)

    with pytest.raises(SQLProposalStateError, match="statement_timeout"):
        service.approve_and_execute(principal=_principal(), proposal_id=proposal.id)

    assert proposals.mark_failed_calls == ["statement_timeout"]


def test_approve_and_execute_maps_unknown_errors_to_execution_error() -> None:
    service, _, proposals, _ = _service(execute_error=RuntimeError("something unexpected"))
    proposal = _create_proposal(service)

    with pytest.raises(SQLProposalStateError, match="execution_error"):
        service.approve_and_execute(principal=_principal(), proposal_id=proposal.id)

    assert proposals.mark_failed_calls == ["execution_error"]


def test_reject_delegates_to_repository() -> None:
    service, _, proposals, _ = _service()
    proposal = _create_proposal(service)

    service.reject(principal=_principal(), proposal_id=proposal.id)

    assert proposals._proposals[proposal.id].status is SQLProposalStatus.REJECTED  # noqa: SLF001


def test_expired_proposal_cannot_be_approved() -> None:
    service, _, proposals, _ = _service()
    proposal = _create_proposal(service)
    stored = proposals._proposals[proposal.id]  # noqa: SLF001
    proposals._proposals[proposal.id] = SQLProposal(  # noqa: SLF001
        **{**stored.__dict__, "expires_at": datetime.now(UTC) - timedelta(seconds=1)}
    )

    with pytest.raises(SQLProposalStateError, match="proposal_expired"):
        service.approve_and_execute(principal=_principal(), proposal_id=proposal.id)


def test_unknown_proposal_id_is_not_silently_accepted() -> None:
    service, *_ = _service()

    with pytest.raises(KeyError):
        service.approve_and_execute(principal=_principal(), proposal_id=uuid4())
