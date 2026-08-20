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

from app.sql.models import (
    CatalogSnapshot,
    SanitizedSQLResult,
    SQLExecutionResult,
    SQLPrincipal,
    ValidatedSQL,
)

_MASK = "***"


class SQLResultPolicy(Protocol):
    def apply(
        self,
        result: SQLExecutionResult,
        principal: SQLPrincipal,
        catalog: CatalogSnapshot,
        validated: ValidatedSQL,
    ) -> SanitizedSQLResult: ...


class DefaultSQLResultPolicy:
    def apply(
        self,
        result: SQLExecutionResult,
        principal: SQLPrincipal,
        catalog: CatalogSnapshot,
        validated: ValidatedSQL,
    ) -> SanitizedSQLResult:
        # "admin" does not imply "may see PII" - that needs its own
        # explicit permission (e.g. sql:view_pii) this release doesn't yet
        # grant to anyone, so every result is masked regardless of
        # principal.is_admin. See the architecture blueprint's result-policy
        # section.
        del principal, catalog
        # Masked by AST-resolved lineage (validated.projection_sensitive),
        # never by the driver-returned column label - an alias like
        # "SELECT c.ssn AS harmless" must still mask, since the returned
        # label "harmless" carries no information about where the value
        # actually came from. See ValidatedSQL.projection_sensitive and
        # app.sql.sql_policy.SQLPolicy.validate_and_rewrite.
        masked_rows = tuple(
            tuple(
                _MASK if is_sensitive else value
                for is_sensitive, value in zip(validated.projection_sensitive, row, strict=True)
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
