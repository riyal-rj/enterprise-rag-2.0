"""Restricted PostgreSQL planning/execution for validated SQL only.

Every call runs against a **separate** ``PostgresConnectionPool`` pointed at
``SQLFeatureSettings.sql_database_url`` - never the application's own
``get_db_pool()`` (see ``app.core.db``) - and every transaction is forced
read-only with bounded statement/lock timeouts before a single statement
runs. This is the layer PostgreSQL itself enforces; the AST policy
(``app.sql.sql_policy``) is a defense-in-depth check ahead of it, not a
replacement for it - see that module's docstring.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

from psycopg2.extensions import connection as PgConnection

from app.core.db import PostgresConnectionPool
from app.sql.models import ExplainAssessment, SQLExecutionResult, SQLPrincipal, ValidatedSQL


class SQLExecutionRejected(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class SQLExecutionLimits:
    statement_timeout_ms: int
    lock_timeout_ms: int
    max_plan_cost: float
    max_plan_rows: int
    max_result_rows: int
    max_result_bytes: int
    max_cell_chars: int

    def __post_init__(self) -> None:
        if self.statement_timeout_ms <= 0:
            raise ValueError("statement_timeout_ms must be positive")
        if self.lock_timeout_ms <= 0:
            raise ValueError("lock_timeout_ms must be positive")
        if self.max_plan_cost <= 0:
            raise ValueError("max_plan_cost must be positive")
        if self.max_plan_rows <= 0:
            raise ValueError("max_plan_rows must be positive")
        if self.max_result_rows <= 0:
            raise ValueError("max_result_rows must be positive")
        if self.max_result_bytes <= 0:
            raise ValueError("max_result_bytes must be positive")
        if self.max_cell_chars <= 0:
            raise ValueError("max_cell_chars must be positive")


class SQLExecutor(Protocol):
    def explain(self, query: ValidatedSQL, principal: SQLPrincipal) -> ExplainAssessment: ...

    def execute(self, query: ValidatedSQL, principal: SQLPrincipal) -> SQLExecutionResult: ...


class UnavailableSQLExecutor:
    """Fail-closed stand-in used when ``SQLFeatureSettings.sql_database_url``
    is unset - no analytics database has been provisioned for this
    deployment yet (see ``scripts/sql/provision_sql_reader_role.sql``).
    Lets ``SQLService``/``QueryOrchestrator`` always be constructible at
    process start (matching every other dynamic RAG Ops dependency, which
    is always built regardless of its feature's current toggle state) while
    guaranteeing any actual proposal/execution attempt fails loudly instead
    of connecting to nothing."""

    def explain(self, query: ValidatedSQL, principal: SQLPrincipal) -> ExplainAssessment:
        del query, principal
        raise SQLExecutionRejected("sql_database_not_configured")

    def execute(self, query: ValidatedSQL, principal: SQLPrincipal) -> SQLExecutionResult:
        del query, principal
        raise SQLExecutionRejected("sql_database_not_configured")


class PostgresReadOnlySQLExecutor:
    """The only component in this codebase that runs generated SQL. Every
    connection it uses gets ``SET TRANSACTION READ ONLY`` plus a bounded
    statement/lock timeout before anything else runs, and every transaction
    ends in ``rollback()`` (a ``SELECT`` needs no commit; rolling back also
    guarantees no session-level ``SET`` state leaks back into the pool)."""

    def __init__(self, *, pool: PostgresConnectionPool, limits: SQLExecutionLimits) -> None:
        self._pool = pool
        self._limits = limits

    def explain(self, query: ValidatedSQL, principal: SQLPrincipal) -> ExplainAssessment:
        with self._pool.connection() as conn:
            try:
                self._prepare_transaction(conn, principal)
                with conn.cursor() as cur:
                    cur.execute(f"EXPLAIN (FORMAT JSON) {query.sql}")
                    row = cur.fetchone()
                    if row is None:
                        raise SQLExecutionRejected("explain_returned_no_plan")
                    raw = row[0]
                    plan_document = raw if isinstance(raw, list) else json.loads(raw)
                    plan = plan_document[0]["Plan"]
                    assessment = ExplainAssessment(
                        total_cost=float(plan.get("Total Cost", 0.0)),
                        plan_rows=int(plan.get("Plan Rows", 0)),
                        plan_width=int(plan.get("Plan Width", 0)),
                    )
                if assessment.total_cost > self._limits.max_plan_cost:
                    raise SQLExecutionRejected("plan_cost_exceeded")
                if assessment.plan_rows > self._limits.max_plan_rows:
                    raise SQLExecutionRejected("plan_rows_exceeded")
                return assessment
            finally:
                conn.rollback()

    def execute(self, query: ValidatedSQL, principal: SQLPrincipal) -> SQLExecutionResult:
        with self._pool.connection() as conn:
            started = time.perf_counter()
            try:
                self._prepare_transaction(conn, principal)
                with conn.cursor() as cur:
                    cur.execute(query.sql)
                    columns = tuple(desc[0] for desc in cur.description or ())
                    fetched = cur.fetchmany(self._limits.max_result_rows + 1)
                    truncated = len(fetched) > self._limits.max_result_rows
                    rows = fetched[: self._limits.max_result_rows]

                bounded: list[tuple[Any, ...]] = []
                total_bytes = 0
                for row in rows:
                    safe_row: list[Any] = []
                    row_over_budget = False
                    for cell in row:
                        value = self._bound_cell(cell)
                        total_bytes += len(str(value).encode("utf-8"))
                        if total_bytes > self._limits.max_result_bytes:
                            truncated = True
                            row_over_budget = True
                            break
                        safe_row.append(value)
                    if row_over_budget:
                        break
                    bounded.append(tuple(safe_row))

                return SQLExecutionResult(
                    columns=columns,
                    rows=tuple(bounded),
                    row_count=len(bounded),
                    truncated=truncated,
                    duration_ms=(time.perf_counter() - started) * 1_000,
                    bytes_returned=total_bytes,
                )
            finally:
                conn.rollback()

    def _prepare_transaction(self, conn: PgConnection, principal: SQLPrincipal) -> None:
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute(
                "SELECT set_config('statement_timeout', %s, true)",
                (f"{self._limits.statement_timeout_ms}ms",),
            )
            cur.execute(
                "SELECT set_config('lock_timeout', %s, true)",
                (f"{self._limits.lock_timeout_ms}ms",),
            )
            cur.execute("SELECT set_config('app.current_user', %s, true)", (principal.username,))
            cur.execute(
                "SELECT set_config('app.tenant_id', %s, true)", (principal.tenant_id or "",)
            )

    def _bound_cell(self, value: Any) -> Any:
        if isinstance(value, str) and len(value) > self._limits.max_cell_chars:
            return value[: self._limits.max_cell_chars] + "…"
        return value
