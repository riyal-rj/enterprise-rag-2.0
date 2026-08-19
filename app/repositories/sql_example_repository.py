"""Approved few-shot SQL examples: curated, versioned, human-reviewed.

Distinct from "every query a user ever approved" - promotion to an example
requires review (see the architecture blueprint's warning against
auto-training on execution history, which risks poisoning future generation
with a query that merely executed successfully rather than one that's
actually correct). No HTTP route exposes ``add`` in this rollout; examples
are seeded directly against this table until an admin approval workflow is
built - the repository exists so that workflow has somewhere to write.
"""

from __future__ import annotations

from typing import Any, Protocol

from psycopg2.extras import Json

from app.core.db import PostgresConnectionPool
from app.sql.models import ApprovedSQLExample

_EXAMPLE_COLUMNS = (
    "id, question, sql_text, sql_fingerprint, catalog_version, "
    "approved_by, approved_at, active, tags"
)


class SQLExampleRepository(Protocol):
    def list_active(self, limit: int) -> list[ApprovedSQLExample]: ...

    def add(
        self,
        *,
        question: str,
        sql: str,
        sql_fingerprint: str,
        catalog_version: int,
        approved_by: str,
        tags: tuple[str, ...] = (),
    ) -> ApprovedSQLExample: ...


class PostgresSQLExampleRepository:
    def __init__(self, pool: PostgresConnectionPool) -> None:
        self._pool = pool

    def list_active(self, limit: int) -> list[ApprovedSQLExample]:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {_EXAMPLE_COLUMNS} FROM sql_approved_examples "
                "WHERE active = TRUE ORDER BY approved_at DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
        return [_row_to_example(row) for row in rows]

    def add(
        self,
        *,
        question: str,
        sql: str,
        sql_fingerprint: str,
        catalog_version: int,
        approved_by: str,
        tags: tuple[str, ...] = (),
    ) -> ApprovedSQLExample:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO sql_approved_examples "
                    "(question, sql_text, sql_fingerprint, catalog_version, approved_by, tags) "
                    "VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (sql_fingerprint) DO UPDATE SET "
                    "question = EXCLUDED.question, active = TRUE "
                    f"RETURNING {_EXAMPLE_COLUMNS}",
                    (
                        question,
                        sql,
                        sql_fingerprint,
                        catalog_version,
                        approved_by,
                        Json(list(tags)),
                    ),
                )
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError("Failed to retrieve inserted SQL example")
            conn.commit()
        return _row_to_example(row)


def _row_to_example(row: tuple[Any, ...]) -> ApprovedSQLExample:
    return ApprovedSQLExample(
        id=row[0],
        question=row[1],
        sql=row[2],
        sql_fingerprint=row[3],
        catalog_version=row[4],
        approved_by=row[5],
        approved_at=row[6],
        active=row[7],
        tags=tuple(row[8] or []),
    )
