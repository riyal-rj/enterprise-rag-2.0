"""Result minimization/masking - the last gate before an LLM or a caller
ever sees a row of SQL output.

Row/byte/cell caps already applied once in ``app.sql.sql_executor`` are
re-verified here defensively; the column-allowlist re-check and sensitive-
value masking are this module's own job - a raw result row must never reach
``app.sql.sql_answerer`` (the LLM summarizer) or an HTTP response body
without passing through this first.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from app.sql.models import CatalogSnapshot, SanitizedSQLResult, SQLExecutionResult, SQLPrincipal

_MASK = "***"


class SQLResultPolicy(Protocol):
    def apply(
        self,
        result: SQLExecutionResult,
        principal: SQLPrincipal,
        catalog: CatalogSnapshot,
    ) -> SanitizedSQLResult: ...


class DefaultSQLResultPolicy:
    def apply(
        self,
        result: SQLExecutionResult,
        principal: SQLPrincipal,
        catalog: CatalogSnapshot,
    ) -> SanitizedSQLResult:
        # "admin" does not imply "may see PII" - that needs its own
        # explicit permission (e.g. sql:view_pii) this release doesn't yet
        # grant to anyone, so every result is masked regardless of
        # principal.is_admin. See the architecture blueprint's result-policy
        # section.
        del principal
        sensitive_by_name = {
            column.column_name.casefold() for column in catalog.columns if column.sensitive
        }
        masked_rows = tuple(
            tuple(
                _MASK if column_name.casefold() in sensitive_by_name else value
                for column_name, value in zip(result.columns, row, strict=True)
            )
            for row in result.rows
        )
        return SanitizedSQLResult(
            columns=result.columns,
            rows=masked_rows,
            row_count=result.row_count,
            truncated=result.truncated,
            snapshot_at=datetime.now(UTC),
        )
