from __future__ import annotations

from app.sql.models import CatalogColumn, CatalogSnapshot, SQLExecutionResult, SQLPrincipal
from app.sql.sql_result_policy import DefaultSQLResultPolicy


def _catalog() -> CatalogSnapshot:
    return CatalogSnapshot(
        version=1,
        columns=(
            CatalogColumn("approved_analytics", "accounts", "id", "bigint", None, False),
            CatalogColumn("approved_analytics", "accounts", "ssn", "text", None, True),
        ),
        relationships=(),
        business_rules=(),
    )


def test_sensitive_columns_are_masked_regardless_of_admin_status() -> None:
    result = SQLExecutionResult(
        columns=("id", "ssn"),
        rows=((1, "123-45-6789"),),
        row_count=1,
        truncated=False,
        duration_ms=1.0,
        bytes_returned=20,
    )
    policy = DefaultSQLResultPolicy()

    sanitized = policy.apply(result, SQLPrincipal(username="admin", is_admin=True), _catalog())

    assert sanitized.rows == ((1, "***"),)


def test_non_sensitive_columns_pass_through_unchanged() -> None:
    result = SQLExecutionResult(
        columns=("id",),
        rows=((1,), (2,)),
        row_count=2,
        truncated=False,
        duration_ms=1.0,
        bytes_returned=10,
    )
    policy = DefaultSQLResultPolicy()

    sanitized = policy.apply(result, SQLPrincipal(username="alice", is_admin=False), _catalog())

    assert sanitized.rows == ((1,), (2,))
    assert sanitized.row_count == 2


def test_truncated_flag_and_row_count_are_preserved() -> None:
    result = SQLExecutionResult(
        columns=("id",),
        rows=((1,),),
        row_count=1,
        truncated=True,
        duration_ms=1.0,
        bytes_returned=5,
    )
    policy = DefaultSQLResultPolicy()

    sanitized = policy.apply(result, SQLPrincipal(username="alice", is_admin=False), _catalog())

    assert sanitized.truncated is True
    assert sanitized.row_count == 1
