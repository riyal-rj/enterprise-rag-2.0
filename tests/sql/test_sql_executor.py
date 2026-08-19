"""Unit tests for PostgresReadOnlySQLExecutor's own logic (cell bounding,
read-only transaction setup) against a fake connection/cursor - no real
Postgres needed. See test_sql_executor_integration.py for the real-database
counterpart, which needs SQL_DATABASE_URL configured."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from app.sql.models import SQLPrincipal, ValidatedSQL
from app.sql.sql_executor import (
    PostgresReadOnlySQLExecutor,
    SQLExecutionLimits,
    SQLExecutionRejected,
    UnavailableSQLExecutor,
)


class _FakeCursor:
    def __init__(
        self, *, description: tuple[tuple[str], ...], rows: list[tuple[object, ...]]
    ) -> None:
        self.description = description
        self._rows = rows
        self.executed: list[str] = []
        self.executed_params: list[object] = []

    def execute(self, sql: str, params: object = None) -> None:
        self.executed.append(sql)
        self.executed_params.append(params)

    def fetchmany(self, size: int) -> list[tuple[object, ...]]:
        return self._rows[:size]

    def fetchone(self) -> tuple[object, ...] | None:
        return self._rows[0] if self._rows else None

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor
        self.autocommit: bool | None = None
        self.rolled_back = False

    def cursor(self) -> _FakeCursor:
        return self._cursor

    def rollback(self) -> None:
        self.rolled_back = True


class _FakePool:
    def __init__(self, conn: _FakeConnection) -> None:
        self._conn = conn

    @contextmanager
    def connection(self):  # noqa: ANN201
        yield self._conn


def _limits(**overrides: object) -> SQLExecutionLimits:
    defaults: dict[str, object] = dict(
        statement_timeout_ms=5_000,
        lock_timeout_ms=500,
        max_plan_cost=100_000.0,
        max_plan_rows=100_000,
        max_result_rows=200,
        max_result_bytes=250_000,
        max_cell_chars=2_000,
    )
    defaults.update(overrides)
    return SQLExecutionLimits(**defaults)  # type: ignore[arg-type]


def _query() -> ValidatedSQL:
    return ValidatedSQL(
        sql="SELECT a.id FROM approved_analytics.accounts a LIMIT 200",
        fingerprint="f" * 64,
        referenced_tables=("approved_analytics.accounts",),
        referenced_columns=("approved_analytics.accounts.id",),
        row_limit=200,
        policy_version="sql-policy-v1",
    )


def _principal() -> SQLPrincipal:
    return SQLPrincipal(username="admin", is_admin=True)


@pytest.mark.parametrize(
    ("value", "max_cell_chars", "expected"),
    [
        ("short", 10, "short"),
        ("a" * 20, 10, "a" * 10 + "…"),
        (42, 10, 42),
        (None, 10, None),
    ],
)
def test_bound_cell(value: object, max_cell_chars: int, expected: object) -> None:
    conn = _FakeConnection(_FakeCursor(description=(), rows=[]))
    executor = PostgresReadOnlySQLExecutor(
        pool=_FakePool(conn), limits=_limits(max_cell_chars=max_cell_chars)
    )

    assert executor._bound_cell(value) == expected


def test_execute_sets_read_only_transaction_and_timeouts_before_running_query() -> None:
    cursor = _FakeCursor(description=(("id",),), rows=[(1,), (2,)])
    conn = _FakeConnection(cursor)
    executor = PostgresReadOnlySQLExecutor(pool=_FakePool(conn), limits=_limits())

    result = executor.execute(_query(), _principal())

    assert cursor.executed[0] == "SET TRANSACTION READ ONLY"
    assert any("statement_timeout" in call for call in cursor.executed[1:3])
    assert result.columns == ("id",)
    assert result.row_count == 2
    assert conn.rolled_back is True


def test_execute_truncates_when_more_rows_than_max_result_rows() -> None:
    cursor = _FakeCursor(description=(("id",),), rows=[(i,) for i in range(5)])
    conn = _FakeConnection(cursor)
    executor = PostgresReadOnlySQLExecutor(pool=_FakePool(conn), limits=_limits(max_result_rows=3))

    result = executor.execute(_query(), _principal())

    assert result.row_count == 3
    assert result.truncated is True


def test_execute_rolls_back_even_when_the_query_raises() -> None:
    class _RaisingCursor(_FakeCursor):
        def execute(self, sql: str, params: object = None) -> None:  # noqa: ARG002
            if "SELECT a.id" in sql:
                raise RuntimeError("boom")
            super().execute(sql, params)

    cursor = _RaisingCursor(description=(), rows=[])
    conn = _FakeConnection(cursor)
    executor = PostgresReadOnlySQLExecutor(pool=_FakePool(conn), limits=_limits())

    with pytest.raises(RuntimeError):
        executor.execute(_query(), _principal())

    assert conn.rolled_back is True


def _explain_cursor(total_cost: float, plan_rows: int) -> _FakeCursor:
    plan_json = [{"Plan": {"Total Cost": total_cost, "Plan Rows": plan_rows, "Plan Width": 8}}]
    return _FakeCursor(description=(), rows=[(plan_json,)])


def test_explain_accepts_a_plan_within_limits() -> None:
    conn = _FakeConnection(_explain_cursor(total_cost=100.0, plan_rows=50))
    executor = PostgresReadOnlySQLExecutor(
        pool=_FakePool(conn), limits=_limits(max_plan_cost=1_000.0, max_plan_rows=1_000)
    )

    assessment = executor.explain(_query(), _principal())

    assert assessment.total_cost == 100.0
    assert assessment.plan_rows == 50
    assert conn.rolled_back is True


def test_explain_rejects_a_plan_that_exceeds_max_cost() -> None:
    conn = _FakeConnection(_explain_cursor(total_cost=999_999.0, plan_rows=1))
    executor = PostgresReadOnlySQLExecutor(
        pool=_FakePool(conn), limits=_limits(max_plan_cost=1_000.0)
    )

    with pytest.raises(SQLExecutionRejected, match="plan_cost_exceeded"):
        executor.explain(_query(), _principal())


def test_explain_rejects_a_plan_that_exceeds_max_rows() -> None:
    conn = _FakeConnection(_explain_cursor(total_cost=1.0, plan_rows=999_999))
    executor = PostgresReadOnlySQLExecutor(
        pool=_FakePool(conn), limits=_limits(max_plan_rows=1_000)
    )

    with pytest.raises(SQLExecutionRejected, match="plan_rows_exceeded"):
        executor.explain(_query(), _principal())


def test_unavailable_executor_fails_closed_on_explain_and_execute() -> None:
    executor = UnavailableSQLExecutor()

    with pytest.raises(SQLExecutionRejected, match="sql_database_not_configured"):
        executor.explain(_query(), _principal())
    with pytest.raises(SQLExecutionRejected, match="sql_database_not_configured"):
        executor.execute(_query(), _principal())
