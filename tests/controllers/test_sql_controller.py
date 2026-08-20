from __future__ import annotations

from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

from app.controllers.sql_controller import SQLController
from app.models.auth_user import AuthenticatedUser
from app.sql.models import (
    SanitizedSQLResult,
    SQLCompletedAnswer,
    SQLPrincipal,
    SQLProposal,
    SQLProposalStatus,
    ValidatedSQL,
)
from app.sql.sql_service import SQLService


def _proposal(status: SQLProposalStatus = SQLProposalStatus.EXECUTED) -> SQLProposal:
    now = datetime.now(UTC)
    return SQLProposal(
        id=uuid4(),
        username="admin",
        conversation_id=1,
        question="how many accounts?",
        sql="SELECT a.id FROM approved_analytics.accounts a LIMIT 10",
        sql_fingerprint="f" * 64,
        referenced_tables=("approved_analytics.accounts",),
        assumptions=(),
        catalog_version=1,
        policy_version="sql-policy-v1",
        status=status,
        expires_at=now,
        created_at=now,
    )


def _completed_answer() -> SQLCompletedAnswer:
    proposal = _proposal()
    validated = ValidatedSQL(
        sql=proposal.sql,
        fingerprint=proposal.sql_fingerprint,
        referenced_tables=proposal.referenced_tables,
        referenced_columns=("approved_analytics.accounts.id",),
        projection_sensitive=(False,),
        row_limit=10,
        policy_version=proposal.policy_version,
    )
    result = SanitizedSQLResult(
        columns=("id",), rows=((1,),), row_count=1, truncated=False, snapshot_at=datetime.now(UTC)
    )
    return SQLCompletedAnswer(
        answer="1 row found.", result=result, proposal=proposal, validated=validated
    )


class _FakeSQLService:
    def __init__(self, *, completed: SQLCompletedAnswer | None = None) -> None:
        self._completed = completed
        self.approve_calls: list[UUID] = []
        self.reject_calls: list[UUID] = []

    def approve_and_execute(
        self, *, principal: SQLPrincipal, proposal_id: UUID
    ) -> SQLCompletedAnswer:
        del principal
        self.approve_calls.append(proposal_id)
        assert self._completed is not None
        return self._completed

    def reject(self, *, principal: SQLPrincipal, proposal_id: UUID) -> None:
        del principal
        self.reject_calls.append(proposal_id)


def _admin() -> AuthenticatedUser:
    return AuthenticatedUser(username="admin", is_admin=True)


def test_approve_returns_completed_response_with_sql_metadata() -> None:
    completed = _completed_answer()
    sql_service = _FakeSQLService(completed=completed)
    controller = SQLController(cast(SQLService, sql_service))

    response = controller.approve(_admin(), completed.proposal.id)

    assert response.status == "completed"
    assert response.answer == "1 row found."
    assert response.metadata.route == "sql"
    assert response.metadata.sql.enabled is True
    assert response.metadata.sql.row_count == 1
    assert response.metadata.sql.normalized_sql == completed.validated.sql
    assert sql_service.approve_calls == [completed.proposal.id]


def test_reject_delegates_to_service() -> None:
    sql_service = _FakeSQLService()
    controller = SQLController(cast(SQLService, sql_service))
    proposal_id = uuid4()

    controller.reject(_admin(), proposal_id)

    assert sql_service.reject_calls == [proposal_id]
