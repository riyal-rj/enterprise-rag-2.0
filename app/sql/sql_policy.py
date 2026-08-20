"""Defense-in-depth SQL AST policy.

sqlglot's parser is deliberately lenient (it's a transpiler, not a security
sandbox - see its own documentation) - this module is one layer of many
(see ``app.sql.sql_executor`` for the authoritative PostgreSQL-side
``EXPLAIN``/read-only-transaction check, and ``scripts/sql/
provision_sql_reader_role.sql`` for the database role/RLS layer). Nothing
here is trusted in isolation.

This initial policy version intentionally supports a small SQL subset:
direct ``SELECT`` over concrete allowlisted tables with fully qualified
columns, bounded joins, approved functions, and a literal ``LIMIT``. CTEs,
subqueries, set operations, and lateral sources are rejected outright rather
than half-supported - their column lineage needs its own test corpus before
they're safe to allow (see the architecture blueprint's policy notes).
Unknown or ambiguous resolution always fails closed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import build_scope

from app.sql.models import CatalogColumn, CatalogSnapshot, SQLPrincipal, ValidatedSQL


class SQLPolicyViolation(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class SQLPolicyConfig:
    policy_version: str
    allowed_schemas: frozenset[str]
    allowed_functions: frozenset[str]
    max_joins: int
    max_rows: int
    require_qualified_tables: bool = True

    def __post_init__(self) -> None:
        if self.max_joins < 0:
            raise ValueError("max_joins must not be negative")
        if self.max_rows < 1:
            raise ValueError("max_rows must be positive")


_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Grant,
)


def _build_qualify_schema(
    columns: tuple[CatalogColumn, ...],
) -> dict[str, dict[str, dict[str, str]]]:
    """``{schema: {table: {column: data_type}}}`` - the shape
    ``sqlglot.optimizer.qualify.qualify`` expects for its ``schema=``
    argument, built from the same approved catalog the rest of this method
    allowlists against. Only ever contains approved objects - qualify()
    cannot resolve (and this policy therefore cannot silently accept) a
    column on a table this schema doesn't mention."""
    schema: dict[str, dict[str, dict[str, str]]] = {}
    for column in columns:
        tables = schema.setdefault(column.schema_name, {})
        table_columns = tables.setdefault(column.table_name, {})
        table_columns[column.column_name] = column.data_type
    return schema


class SQLPolicy:
    def __init__(self, config: SQLPolicyConfig) -> None:
        self._config = config

    @property
    def version(self) -> str:
        return self._config.policy_version

    def validate_and_rewrite(
        self,
        sql: str,
        *,
        catalog: CatalogSnapshot,
        principal: SQLPrincipal,
    ) -> ValidatedSQL:
        del principal  # every principal validated against the same catalog in this release
        if "--" in sql or "/*" in sql or "*/" in sql:
            raise SQLPolicyViolation("comments_forbidden")
        if ";" in sql.rstrip().rstrip(";"):
            raise SQLPolicyViolation("multiple_statements")

        try:
            statements = sqlglot.parse(sql, read="postgres")
        except sqlglot.errors.ParseError as exc:
            raise SQLPolicyViolation("parse_error") from exc
        statements = [statement for statement in statements if statement is not None]
        if len(statements) != 1:
            raise SQLPolicyViolation("multiple_statements")

        tree = statements[0]
        if not isinstance(tree, exp.Select):
            raise SQLPolicyViolation("select_only")
        for node_type in _FORBIDDEN_NODES:
            if tree.find(node_type) is not None:
                raise SQLPolicyViolation("forbidden_statement")
        if tree.find(exp.Into) is not None:
            raise SQLPolicyViolation("select_into_forbidden")
        if tree.find(exp.Lock) is not None:
            raise SQLPolicyViolation("locking_forbidden")
        if tree.find(exp.Star) is not None:
            raise SQLPolicyViolation("select_star_forbidden")
        if tree.find(exp.CTE) is not None or tree.args.get("with") is not None:
            raise SQLPolicyViolation("cte_forbidden_initial_release")
        if tree.find(exp.Subquery) is not None:
            raise SQLPolicyViolation("subquery_forbidden_initial_release")
        if tree.find(exp.Union) is not None or tree.find(exp.SetOperation) is not None:
            raise SQLPolicyViolation("set_operation_forbidden_initial_release")
        if tree.find(exp.Lateral) is not None:
            raise SQLPolicyViolation("lateral_forbidden_initial_release")

        joins = list(tree.find_all(exp.Join))
        if len(joins) > self._config.max_joins:
            raise SQLPolicyViolation("too_many_joins")

        allowed_tables = {
            (column.schema_name.casefold(), column.table_name.casefold())
            for column in catalog.columns
        }
        referenced_tables: set[str] = set()
        for table in tree.find_all(exp.Table):
            schema_name = (table.db or "").casefold()
            table_name = table.name.casefold()
            if self._config.require_qualified_tables and not schema_name:
                raise SQLPolicyViolation("unqualified_table")
            if schema_name not in self._config.allowed_schemas:
                raise SQLPolicyViolation("schema_not_allowed")
            if (schema_name, table_name) not in allowed_tables:
                raise SQLPolicyViolation("table_not_allowed")
            referenced_tables.add(f"{schema_name}.{table_name}")

        # Generators (Vanna included) very often omit table qualifiers on
        # single-table queries ("balance" instead of "accounts.balance") -
        # ordinary, unambiguous SQL that would otherwise fail closed here
        # for no security reason. Resolve every column against the approved
        # catalog schema instead of requiring the generator to have written
        # it explicitly; an unresolvable/ambiguous column still fails closed
        # (validate_qualify_columns=True raises rather than guessing).
        try:
            tree = qualify(
                tree,
                dialect="postgres",
                schema=_build_qualify_schema(catalog.columns),
                validate_qualify_columns=True,
                quote_identifiers=False,
                identify=False,
            )
        except sqlglot.errors.SqlglotError as exc:
            raise SQLPolicyViolation("column_qualification_failed") from exc
        if not isinstance(tree, exp.Select):
            raise SQLPolicyViolation("select_only")

        allowed_columns = {
            (
                column.schema_name.casefold(),
                column.table_name.casefold(),
                column.column_name.casefold(),
            )
            for column in catalog.columns
        }
        referenced_columns: set[str] = set()
        root_scope = build_scope(tree)
        if root_scope is None:
            raise SQLPolicyViolation("scope_resolution_failed")
        for current_scope in root_scope.traverse():
            sources_by_alias = {
                alias.casefold(): source for alias, source in current_scope.sources.items()
            }
            for column in current_scope.columns:
                alias = column.table.casefold()
                if not alias:
                    # qualify() above should have resolved every column - an
                    # empty alias here means resolution silently failed to
                    # attach one rather than raising, which fails closed
                    # rather than guessing which table it meant.
                    raise SQLPolicyViolation("unqualified_column")
                source = sources_by_alias.get(alias)
                if not isinstance(source, exp.Table):
                    # CTE/subquery sources were rejected above; anything
                    # remaining that isn't a concrete table fails closed.
                    raise SQLPolicyViolation("column_source_not_concrete")
                schema_name = (source.db or "").casefold()
                table_name = source.name.casefold()
                column_name = column.name.casefold()
                if (schema_name, table_name, column_name) not in allowed_columns:
                    raise SQLPolicyViolation("column_not_allowed")
                referenced_columns.add(f"{schema_name}.{table_name}.{column_name}")

        # Positional output-column lineage for result masking (see
        # ValidatedSQL.projection_sensitive's docstring). Deliberately
        # separate from the referenced_columns loop above: that loop tracks
        # every column touched anywhere (WHERE/JOIN/SELECT) for the table/
        # column allowlist, while this tracks, in SELECT-list order,
        # whether each *output* column's value can carry a sensitive
        # source value - a bare column reference resolves directly; any
        # other expression (function, arithmetic, concatenation, ...) is
        # sensitive if any leaf column it reads from is sensitive. This
        # errs conservative (e.g. COUNT(ssn) is masked even though a count
        # isn't itself PII) rather than trying to reason about which
        # functions "launder" their sensitive input.
        sensitive_lookup = {
            (
                column.schema_name.casefold(),
                column.table_name.casefold(),
                column.column_name.casefold(),
            ): column.sensitive
            for column in catalog.columns
        }
        root_sources_by_alias = {
            alias.casefold(): source for alias, source in root_scope.sources.items()
        }

        def _column_is_sensitive(column: exp.Column) -> bool:
            alias = column.table.casefold()
            if not alias:
                raise SQLPolicyViolation("unqualified_column")
            source = root_sources_by_alias.get(alias)
            if not isinstance(source, exp.Table):
                raise SQLPolicyViolation("column_source_not_concrete")
            schema_name = (source.db or "").casefold()
            table_name = source.name.casefold()
            column_name = column.name.casefold()
            return sensitive_lookup.get((schema_name, table_name, column_name), False)

        projection_sensitive = tuple(
            any(_column_is_sensitive(column) for column in projection.find_all(exp.Column))
            for projection in tree.selects
        )

        for function in tree.find_all(exp.Func):
            if isinstance(function, exp.Connector):
                # exp.And/exp.Or are Func subclasses in sqlglot's hierarchy
                # (Connector -> Binary -> Func) purely for its own internal
                # consistency - they're boolean connectives, not callable
                # functions, and every WHERE clause with more than one
                # predicate would otherwise need "and"/"or" allowlisted
                # alongside actual functions.
                continue
            name = (function.sql_name() or "").casefold()
            if name and name not in self._config.allowed_functions:
                raise SQLPolicyViolation("function_not_allowed")

        row_limit = self._rewrite_limit(tree)

        normalized = tree.sql(dialect="postgres", pretty=False)
        fingerprint = hashlib.sha256(
            f"{self.version}\x00{catalog.version}\x00{normalized}".encode()
        ).hexdigest()
        return ValidatedSQL(
            sql=normalized,
            fingerprint=fingerprint,
            referenced_tables=tuple(sorted(referenced_tables)),
            referenced_columns=tuple(sorted(referenced_columns)),
            projection_sensitive=projection_sensitive,
            row_limit=row_limit,
            policy_version=self.version,
        )

    def _rewrite_limit(self, tree: exp.Select) -> int:
        limit = tree.args.get("limit")
        if limit is None:
            row_limit = self._config.max_rows
            tree.set("limit", exp.Limit(expression=exp.Literal.number(row_limit)))
            return row_limit

        expression = limit.expression
        if not isinstance(expression, exp.Literal) or not expression.is_int:
            raise SQLPolicyViolation("dynamic_limit_forbidden")
        requested = int(expression.this)
        if requested <= 0:
            raise SQLPolicyViolation("non_positive_limit")
        row_limit = min(requested, self._config.max_rows)
        tree.set("limit", exp.Limit(expression=exp.Literal.number(row_limit)))
        return row_limit
