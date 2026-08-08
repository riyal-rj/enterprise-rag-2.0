"""Pooled Postgres connection management.

Replaces the original per-request ``psycopg2.connect(...)`` with a bounded
connection pool, so request volume can't exhaust the database's connection
limit.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from psycopg2 import pool as pg_pool
from psycopg2.extensions import connection as PgConnection


class PostgresConnectionPool:
    """Thin wrapper around ``psycopg2.pool.SimpleConnectionPool``."""

    def __init__(self, 
                 dsn: str, 
                 minconn: int = 1, 
                 maxconn: int = 10) -> None:
        self._pool: pg_pool.SimpleConnectionPool = pg_pool.SimpleConnectionPool(
            minconn, 
            maxconn, 
            dsn
        )

    @contextmanager
    def connection(self) -> Iterator[PgConnection]:
        """Check out a connection for the duration of the ``with`` block."""
        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    def close(self) -> None:
        """Close every pooled connection. Call once, on app shutdown."""
        self._pool.closeall()
