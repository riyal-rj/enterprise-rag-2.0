"""Security corpus for SQLPolicy - see the Text-to-SQL architecture
blueprint's required security test corpus. Every case here must be
REJECTED; nothing here proves a query is safe to run, only that the AST
layer doesn't wave through the specific attack it names (see
app.sql.sql_policy's module docstring on defense in depth)."""

from __future__ import annotations

import pytest

from app.sql.models import CatalogColumn, CatalogSnapshot, SQLPrincipal
from app.sql.sql_policy import SQLPolicy, SQLPolicyConfig, SQLPolicyViolation


@pytest.fixture
def catalog() -> CatalogSnapshot:
    return CatalogSnapshot(
        version=1,
        columns=(
            CatalogColumn("approved_analytics", "accounts", "id", "bigint", None, False),
            CatalogColumn(
                "approved_analytics", "accounts", "opened_at", "timestamptz", None, False
            ),
            CatalogColumn("approved_analytics", "accounts", "customer_id", "bigint", None, False),
            CatalogColumn("approved_analytics", "accounts", "ssn", "text", None, True),
        ),
        relationships=(),
        business_rules=(),
    )


@pytest.fixture
def policy() -> SQLPolicy:
    return SQLPolicy(
        SQLPolicyConfig(
            policy_version="sql-policy-v1",
            allowed_schemas=frozenset({"approved_analytics"}),
            allowed_functions=frozenset({"count", "sum", "avg", "min", "max", "current_timestamp"}),
            max_joins=2,
            max_rows=200,
        )
    )


@pytest.fixture
def principal() -> SQLPrincipal:
    return SQLPrincipal(username="admin", is_admin=True)


def test_policy_config_rejects_negative_max_joins() -> None:
    with pytest.raises(ValueError, match="max_joins"):
        SQLPolicyConfig(
            policy_version="v1",
            allowed_schemas=frozenset(),
            allowed_functions=frozenset(),
            max_joins=-1,
            max_rows=10,
        )


def test_policy_config_rejects_non_positive_max_rows() -> None:
    with pytest.raises(ValueError, match="max_rows"):
        SQLPolicyConfig(
            policy_version="v1",
            allowed_schemas=frozenset(),
            allowed_functions=frozenset(),
            max_joins=1,
            max_rows=0,
        )


def test_valid_select_is_accepted_and_gets_a_limit_and_fingerprint(
    policy: SQLPolicy, catalog: CatalogSnapshot, principal: SQLPrincipal
) -> None:
    sql = "SELECT a.id, a.opened_at FROM approved_analytics.accounts a WHERE a.customer_id = 5"

    result = policy.validate_and_rewrite(sql, catalog=catalog, principal=principal)

    assert "LIMIT 200" in result.sql
    assert result.referenced_tables == ("approved_analytics.accounts",)
    assert result.referenced_columns == (
        "approved_analytics.accounts.customer_id",
        "approved_analytics.accounts.id",
        "approved_analytics.accounts.opened_at",
    )
    assert len(result.fingerprint) == 64


def test_fingerprint_changes_when_catalog_version_changes(
    policy: SQLPolicy, catalog: CatalogSnapshot, principal: SQLPrincipal
) -> None:
    sql = "SELECT a.id FROM approved_analytics.accounts a"
    first = policy.validate_and_rewrite(sql, catalog=catalog, principal=principal)

    bumped = CatalogSnapshot(
        version=2, columns=catalog.columns, relationships=(), business_rules=()
    )
    second = policy.validate_and_rewrite(sql, catalog=bumped, principal=principal)

    assert first.fingerprint != second.fingerprint


def test_oversized_limit_is_clamped_to_max_rows(
    policy: SQLPolicy, catalog: CatalogSnapshot, principal: SQLPrincipal
) -> None:
    sql = "SELECT a.id FROM approved_analytics.accounts a LIMIT 999999"

    result = policy.validate_and_rewrite(sql, catalog=catalog, principal=principal)

    assert result.row_limit == 200
    assert "LIMIT 200" in result.sql


def test_smaller_explicit_limit_is_preserved(
    policy: SQLPolicy, catalog: CatalogSnapshot, principal: SQLPrincipal
) -> None:
    sql = "SELECT a.id FROM approved_analytics.accounts a LIMIT 10"

    result = policy.validate_and_rewrite(sql, catalog=catalog, principal=principal)

    assert result.row_limit == 10


_SECURITY_CORPUS: dict[str, str] = {
    "insert": "INSERT INTO approved_analytics.accounts (id) VALUES (1)",
    "update": "UPDATE approved_analytics.accounts SET id = 1",
    "delete": "DELETE FROM approved_analytics.accounts",
    "drop": "DROP TABLE approved_analytics.accounts",
    "alter": "ALTER TABLE approved_analytics.accounts ADD COLUMN x int",
    "grant": "GRANT SELECT ON approved_analytics.accounts TO public",
    "copy": "COPY approved_analytics.accounts TO '/tmp/out.csv'",
    "call": "CALL some_proc()",
    "do_block": "DO $$ BEGIN NULL; END $$",
    "stacked_statements": (
        "SELECT a.id FROM approved_analytics.accounts a; DROP TABLE approved_analytics.accounts;"
    ),
    "line_comment": "SELECT a.id FROM approved_analytics.accounts a -- sneaky comment",
    "block_comment": "SELECT a.id FROM approved_analytics.accounts a /* sneaky */",
    "select_star": "SELECT * FROM approved_analytics.accounts a",
    "select_into": "SELECT a.id INTO TEMP foo FROM approved_analytics.accounts a",
    "for_update": "SELECT a.id FROM approved_analytics.accounts a FOR UPDATE",
    "unqualified_table": "SELECT id FROM accounts",
    "unqualified_column": (
        "SELECT id FROM approved_analytics.accounts a JOIN approved_analytics.accounts b "
        "ON a.id = b.id"
    ),
    "schema_not_allowed": "SELECT a.id FROM public.accounts a",
    "table_not_allowed": "SELECT a.id FROM approved_analytics.other_table a",
    "column_not_allowed": "SELECT a.secret_column FROM approved_analytics.accounts a",
    "sensitive_column_still_allowlisted_but_dangerous_func": "SELECT pg_sleep(a.id) FROM approved_analytics.accounts a",
    "cte": ("WITH x AS (SELECT id FROM approved_analytics.accounts) SELECT id FROM x"),
    "subquery": ("SELECT id FROM (SELECT id FROM approved_analytics.accounts) sub"),
    "union": (
        "SELECT a.id FROM approved_analytics.accounts a "
        "UNION SELECT a.id FROM approved_analytics.accounts a"
    ),
    "dynamic_limit": "SELECT a.id FROM approved_analytics.accounts a LIMIT (SELECT 5)",
    "too_many_joins": (
        "SELECT a.id FROM approved_analytics.accounts a "
        "JOIN approved_analytics.accounts b ON a.id = b.id "
        "JOIN approved_analytics.accounts c ON b.id = c.id "
        "JOIN approved_analytics.accounts d ON c.id = d.id"
    ),
}


@pytest.mark.parametrize("sql", _SECURITY_CORPUS.values(), ids=list(_SECURITY_CORPUS.keys()))
def test_security_corpus_is_rejected(
    sql: str, policy: SQLPolicy, catalog: CatalogSnapshot, principal: SQLPrincipal
) -> None:
    with pytest.raises(SQLPolicyViolation):
        policy.validate_and_rewrite(sql, catalog=catalog, principal=principal)


def test_unqualified_columns_on_a_single_table_query_are_auto_resolved(
    policy: SQLPolicy, catalog: CatalogSnapshot, principal: SQLPrincipal
) -> None:
    """Regression: generators (Vanna included) very often omit table
    qualifiers on unambiguous single-table queries - this must resolve via
    the catalog schema, not fail closed just because the generator didn't
    spell out "accounts.balance"."""
    sql = "SELECT AVG(customer_id) FROM approved_analytics.accounts"

    result = policy.validate_and_rewrite(sql, catalog=catalog, principal=principal)

    assert "accounts.customer_id" in result.sql
    assert result.referenced_columns == ("approved_analytics.accounts.customer_id",)


def test_and_in_where_clause_is_not_treated_as_a_disallowed_function(
    policy: SQLPolicy, catalog: CatalogSnapshot, principal: SQLPrincipal
) -> None:
    """Regression: sqlglot's exp.And/exp.Or are Func subclasses internally
    (Connector -> Binary -> Func) even though they're boolean connectives,
    not callable functions - a WHERE clause with more than one predicate
    must not require "and"/"or" to be allowlisted as if they were
    pg_sleep-style functions."""
    sql = "SELECT a.id FROM approved_analytics.accounts a WHERE a.customer_id = 1 AND a.id > 0"

    result = policy.validate_and_rewrite(sql, catalog=catalog, principal=principal)

    assert "AND" in result.sql.upper()


def test_current_timestamp_and_interval_arithmetic_is_accepted(
    policy: SQLPolicy, catalog: CatalogSnapshot, principal: SQLPrincipal
) -> None:
    """Regression: "opened in the last year"-style questions need
    NOW()/CURRENT_TIMESTAMP minus an INTERVAL - both must be usable without
    the generator needing to know they're on an allowlist."""
    sql = (
        "SELECT a.id FROM approved_analytics.accounts a "
        "WHERE a.opened_at >= NOW() - INTERVAL '1 year'"
    )

    result = policy.validate_and_rewrite(sql, catalog=catalog, principal=principal)

    assert "CURRENT_TIMESTAMP" in result.sql.upper()


def test_multiple_statements_via_trailing_semicolon_is_fine(
    policy: SQLPolicy, catalog: CatalogSnapshot, principal: SQLPrincipal
) -> None:
    """A single trailing semicolon (not a second statement) must not be
    rejected as 'multiple statements' - only an actual second statement
    should trip that check."""
    sql = "SELECT a.id FROM approved_analytics.accounts a;"

    result = policy.validate_and_rewrite(sql, catalog=catalog, principal=principal)

    assert result.referenced_tables == ("approved_analytics.accounts",)
