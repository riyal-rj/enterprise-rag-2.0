"""SQL proposal/approval orchestration - the one place that ties generation,
policy, execution, and persistence together.

``propose()`` never executes anything; it only produces an immutable,
expiring, fingerprint-bound proposal a human must separately approve.
``approve_and_execute()`` is the only path that runs a query, and it
revalidates everything (catalog version, AST policy, fingerprint) under the
proposal's row lock before doing so - approval is bound to a specific
proposal, not "run whatever the model would generate now" (see
``app.repositories.sql_proposal_repository.PostgresSQLProposalRepository
.lock_for_execution``).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import psycopg2

from app.core.exceptions import SQLGenerationFailedError, SQLProposalStateError
from app.repositories.sql_example_repository import SQLExampleRepository
from app.repositories.sql_proposal_repository import SQLProposalRepository
from app.sql.catalog import StaticSQLCatalog
from app.sql.models import (
    SQLCompletedAnswer,
    SQLPrincipal,
    SQLProposal,
    SQLProposalStatus,
)
from app.sql.sql_answerer import SQLAnswerer
from app.sql.sql_executor import SQLExecutionRejected, SQLExecutor
from app.sql.sql_generator import VannaSQLGenerator
from app.sql.sql_policy import SQLPolicy, SQLPolicyViolation
from app.sql.sql_result_policy import SQLResultPolicy

logger = logging.getLogger(__name__)


def _map_execution_error(exc: Exception) -> str:
    """Sanitized error code only - never the raw driver message (which can
    include table/column names, literal values, or query text)."""
    if isinstance(exc, psycopg2.errors.QueryCanceled):
        return "statement_timeout"
    if isinstance(exc, psycopg2.errors.LockNotAvailable):
        return "lock_timeout"
    if isinstance(exc, psycopg2.OperationalError):
        return "connection_error"
    return "execution_error"


class SQLService:
    def __init__(
        self,
        *,
        catalog: StaticSQLCatalog,
        generator: VannaSQLGenerator,
        policy: SQLPolicy,
        executor: SQLExecutor,
        result_policy: SQLResultPolicy,
        answerer: SQLAnswerer,
        proposals: SQLProposalRepository,
        examples: SQLExampleRepository,
        proposal_ttl_seconds: int,
        max_generation_attempts: int,
        max_examples: int,
    ) -> None:
        self._catalog = catalog
        self._generator = generator
        self._policy = policy
        self._executor = executor
        self._result_policy = result_policy
        self._answerer = answerer
        self._proposals = proposals
        self._examples = examples
        self._proposal_ttl_seconds = proposal_ttl_seconds
        self._max_generation_attempts = max_generation_attempts
        self._max_examples = max_examples

    def propose(
        self, *, principal: SQLPrincipal, conversation_id: int, question: str
    ) -> SQLProposal:
        catalog = self._catalog.current(principal)
        self._generator.sync_catalog(self._catalog, catalog)
        self._generator.sync_examples(self._examples.list_active(self._max_examples))

        correction_code: str | None = None
        last_error_code = "no_generation_attempted"
        validated = None
        for _attempt in range(self._max_generation_attempts):
            generation = self._generator.generate(
                question=question,
                principal=principal,
                catalog=catalog,
                rendered_catalog="",
                examples=[],
                correction_code=correction_code,
            )
            try:
                validated = self._policy.validate_and_rewrite(
                    generation.sql, catalog=catalog, principal=principal
                )
                self._executor.explain(validated, principal)
            except (SQLPolicyViolation, SQLExecutionRejected) as exc:
                logger.info(
                    "sql.generation_attempt_rejected",
                    extra={"username": principal.username, "code": exc.code},
                )
                correction_code = exc.code
                last_error_code = exc.code
                validated = None
                continue
            break

        if validated is None:
            raise SQLGenerationFailedError(last_error_code)

        now = datetime.now(UTC)
        proposal = SQLProposal(
            id=uuid4(),
            username=principal.username,
            conversation_id=conversation_id,
            question=question,
            sql=validated.sql,
            sql_fingerprint=validated.fingerprint,
            referenced_tables=validated.referenced_tables,
            assumptions=(),
            catalog_version=catalog.version,
            policy_version=validated.policy_version,
            status=SQLProposalStatus.PROPOSED,
            expires_at=now + timedelta(seconds=self._proposal_ttl_seconds),
            created_at=now,
        )
        return self._proposals.create(proposal)

    def approve_and_execute(
        self, *, principal: SQLPrincipal, proposal_id: UUID
    ) -> SQLCompletedAnswer:
        proposal = self._proposals.lock_for_execution(
            proposal_id, principal.username, datetime.now(UTC)
        )
        catalog = self._catalog.current(principal)
        if catalog.version != proposal.catalog_version:
            self._proposals.mark_failed(proposal.id, error_code="catalog_changed")
            raise SQLProposalStateError("catalog_changed_regenerate_proposal")

        try:
            validated = self._policy.validate_and_rewrite(
                proposal.sql, catalog=catalog, principal=principal
            )
        except SQLPolicyViolation as exc:
            self._proposals.mark_failed(proposal.id, error_code=exc.code)
            raise SQLProposalStateError(exc.code) from exc

        if validated.fingerprint != proposal.sql_fingerprint:
            self._proposals.mark_failed(proposal.id, error_code="fingerprint_mismatch")
            raise SQLProposalStateError("fingerprint_mismatch")

        try:
            self._executor.explain(validated, principal)
            raw = self._executor.execute(validated, principal)
        except SQLExecutionRejected as exc:
            self._proposals.mark_failed(proposal.id, error_code=exc.code)
            raise SQLProposalStateError(exc.code) from exc
        except Exception as exc:  # noqa: BLE001 - mapped to a sanitized code below, then re-raised
            code = _map_execution_error(exc)
            logger.warning(
                "sql.execution_failed",
                extra={
                    "username": principal.username,
                    "code": code,
                    "error_type": type(exc).__name__,
                },
            )
            self._proposals.mark_failed(proposal.id, error_code=code)
            raise SQLProposalStateError(code) from exc

        safe = self._result_policy.apply(raw, principal, catalog)
        answer_text = self._answerer.answer(
            question=proposal.question, query=validated, result=safe
        )

        self._proposals.mark_executed(
            proposal.id, row_count=safe.row_count, execution_ms=raw.duration_ms
        )
        executed_proposal = SQLProposal(
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
            status=SQLProposalStatus.EXECUTED,
            expires_at=proposal.expires_at,
            created_at=proposal.created_at,
        )
        return SQLCompletedAnswer(
            answer=answer_text, result=safe, proposal=executed_proposal, validated=validated
        )

    def reject(self, *, principal: SQLPrincipal, proposal_id: UUID) -> None:
        self._proposals.reject(proposal_id, principal.username)
