"""Text-to-SQL domain models: no provider or HTTP dependencies.

Kept dependency-free (no sqlglot, no psycopg2, no Vanna) so every other
``app.sql``/``app.query_orchestration`` module can import these types without
pulling in a concrete implementation - the same separation
``app.models.rag_ops`` keeps from ``app.repositories.rag_ops_repository``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from app.models.auth_user import AuthenticatedUser


class QueryRoute(StrEnum):
    RAG = "rag"
    SQL = "sql"
    HYBRID_RAG_SQL = "hybrid_rag_sql"
    REJECT = "reject"


class SQLProposalStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    EXECUTING = "executing"
    EXECUTED = "executed"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass(frozen=True)
class RouteDecision:
    route: QueryRoute
    confidence: float
    reason_code: str


@dataclass(frozen=True)
class SQLPrincipal:
    """Capability scope passed to the generator/policy/executor - never the
    raw ``AuthenticatedUser``. ``tenant_id``/``permissions`` stay empty/None
    until the auth system carries richer claims (see
    :func:`to_sql_principal`'s docstring); this release is admin-only, so
    only ``is_admin`` is actually load-bearing today."""

    username: str
    is_admin: bool
    tenant_id: str | None = None
    permissions: frozenset[str] = field(default_factory=frozenset)


def to_sql_principal(user: AuthenticatedUser) -> SQLPrincipal:
    """Map the app's existing auth identity onto the SQL module's principal.

    Deliberately thin: ``AuthenticatedUser`` only carries ``username``/
    ``is_admin`` today (see ``app.models.auth_user``), which is sufficient
    for this admin-only, approval-required rollout. Widening
    ``AuthenticatedUser`` with tenant/entitlement claims is required before
    *general*-user SQL access, not for this phase - see the architecture
    blueprint's Non-negotiable controls / general-user rollout gate.
    """
    return SQLPrincipal(username=user.username, is_admin=user.is_admin)


@dataclass(frozen=True)
class CatalogColumn:
    schema_name: str
    table_name: str
    column_name: str
    data_type: str
    description: str | None
    sensitive: bool


@dataclass(frozen=True)
class CatalogSnapshot:
    version: int
    columns: tuple[CatalogColumn, ...]
    relationships: tuple[str, ...]
    business_rules: tuple[str, ...]


@dataclass(frozen=True)
class SQLGeneration:
    sql: str
    tables: tuple[str, ...]
    assumptions: tuple[str, ...]
    usage_tokens: int
    duration_ms: float


@dataclass(frozen=True)
class ValidatedSQL:
    sql: str
    fingerprint: str
    referenced_tables: tuple[str, ...]
    referenced_columns: tuple[str, ...]
    row_limit: int
    policy_version: str


@dataclass(frozen=True)
class ExplainAssessment:
    total_cost: float
    plan_rows: int
    plan_width: int


@dataclass(frozen=True)
class SQLProposal:
    id: UUID
    username: str
    conversation_id: int
    question: str
    sql: str
    sql_fingerprint: str
    referenced_tables: tuple[str, ...]
    assumptions: tuple[str, ...]
    catalog_version: int
    policy_version: str
    status: SQLProposalStatus
    expires_at: datetime
    created_at: datetime


@dataclass(frozen=True)
class SQLExecutionResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    row_count: int
    truncated: bool
    duration_ms: float
    bytes_returned: int


@dataclass(frozen=True)
class SanitizedSQLResult:
    """Result rows after :class:`app.sql.sql_result_policy.SQLResultPolicy`
    has re-checked the column allowlist, masked sensitive values, and
    applied row/byte/cell minimization - the only row shape ever handed to
    the LLM summarizer or returned to a caller."""

    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    row_count: int
    truncated: bool
    snapshot_at: datetime


@dataclass(frozen=True)
class SQLCompletedAnswer:
    """Result of :meth:`app.sql.sql_service.SQLService.approve_and_execute` -
    what a caller (the approval HTTP route, ``QueryOrchestrator``) actually
    gets back once a proposal has run."""

    answer: str
    result: SanitizedSQLResult
    proposal: SQLProposal
    validated: ValidatedSQL


@dataclass(frozen=True)
class ApprovedSQLExample:
    """A curated, human-reviewed question/SQL pair used to steer generation.

    Distinct from an executed proposal: promotion to an example requires
    review, not just a user approving a query once (see the architecture
    blueprint's warning against auto-training on every executed query).
    """

    id: int
    question: str
    sql: str
    sql_fingerprint: str
    catalog_version: int
    approved_by: str
    approved_at: datetime
    active: bool
    tags: tuple[str, ...] = ()
