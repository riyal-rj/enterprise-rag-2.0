"""Allowlisted SQL catalog: curated, versioned, never live-introspected.

The generator only ever sees approved schema/table/column objects - it never
introspects the analytics database at request time (see the architecture
blueprint's catalog principles). ``StaticSQLCatalog`` reads that allowlist
from a small JSON definition file; the version number it stamps onto every
snapshot comes from ``sql_catalog_state`` (app DB, migration 011), bumped by
an out-of-band admin/background refresh, never by a live request.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from app.core.db import PostgresConnectionPool
from app.sql.models import CatalogColumn, CatalogSnapshot, SQLPrincipal

_EMPTY_DEFINITION: dict[str, Any] = {"columns": [], "relationships": [], "business_rules": []}


class SQLCatalog(Protocol):
    def current(self, principal: SQLPrincipal) -> CatalogSnapshot: ...

    def render_for_model(
        self, snapshot: CatalogSnapshot, question: str, *, max_chars: int
    ) -> str: ...


class StaticSQLCatalog:
    """Reads the approved table/column allowlist from ``definition_path``.

    Every principal currently sees the same snapshot - no per-tenant/role
    column filtering. That's acceptable for this admin-only rollout;
    principal-scoped catalog filtering is future work alongside richer
    ``AuthenticatedUser`` claims (see ``app.sql.models.to_sql_principal``).
    """

    def __init__(self, *, pool: PostgresConnectionPool, definition_path: Path) -> None:
        self._pool = pool
        self._definition_path = definition_path

    def current(self, principal: SQLPrincipal) -> CatalogSnapshot:
        del principal  # every principal sees the same snapshot in this release
        definition = self._load_definition()
        return CatalogSnapshot(
            version=self._read_version(),
            columns=tuple(
                CatalogColumn(
                    schema_name=item["schema_name"],
                    table_name=item["table_name"],
                    column_name=item["column_name"],
                    data_type=item["data_type"],
                    description=item.get("description"),
                    sensitive=bool(item.get("sensitive", False)),
                )
                for item in definition.get("columns", [])
            ),
            relationships=tuple(definition.get("relationships", [])),
            business_rules=tuple(definition.get("business_rules", [])),
        )

    def render_for_model(self, snapshot: CatalogSnapshot, question: str, *, max_chars: int) -> str:
        del question  # every approved table is small enough to render whole in this release
        tables: dict[tuple[str, str], list[CatalogColumn]] = {}
        for column in snapshot.columns:
            tables.setdefault((column.schema_name, column.table_name), []).append(column)

        lines: list[str] = []
        for (schema_name, table_name), columns in tables.items():
            lines.append(f"TABLE {schema_name}.{table_name}")
            for column in columns:
                marker = " [sensitive]" if column.sensitive else ""
                description = f" -- {column.description}" if column.description else ""
                lines.append(f"  {column.column_name} {column.data_type}{marker}{description}")
        if snapshot.relationships:
            lines.append("RELATIONSHIPS")
            lines.extend(f"  {relationship}" for relationship in snapshot.relationships)
        if snapshot.business_rules:
            lines.append("BUSINESS RULES")
            lines.extend(f"  {rule}" for rule in snapshot.business_rules)
        return "\n".join(lines)[:max_chars]

    def ddl_statements(self, snapshot: CatalogSnapshot) -> list[str]:
        """One ``CREATE TABLE`` statement per approved table, for training
        the generator (see ``app.sql.sql_generator.VannaSQLGenerator``).
        Column order follows the definition file; types are copied as-is
        from ``data_type`` (already Postgres-native strings, e.g.
        ``numeric(12,2)``) since this never runs against a real database."""
        tables: dict[tuple[str, str], list[CatalogColumn]] = {}
        for column in snapshot.columns:
            tables.setdefault((column.schema_name, column.table_name), []).append(column)
        statements: list[str] = []
        for (schema_name, table_name), columns in tables.items():
            column_lines = ",\n".join(
                f"    {column.column_name} {column.data_type}" for column in columns
            )
            statements.append(f"CREATE TABLE {schema_name}.{table_name} (\n{column_lines}\n);")
        return statements

    def _load_definition(self) -> dict[str, Any]:
        if not self._definition_path.exists():
            return _EMPTY_DEFINITION
        with self._definition_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)  # type: ignore[no-any-return]

    def _read_version(self) -> int:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT version FROM sql_catalog_state WHERE singleton = TRUE")
            row = cur.fetchone()
        if row is None:
            raise RuntimeError(
                "sql_catalog_state has no row - run migrations "
                "(app/seed/migrations/011_create_text_to_sql.sql)"
            )
        return row[0]  # type: ignore[no-any-return]
