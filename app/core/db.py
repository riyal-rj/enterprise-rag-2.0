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
    """Thin wrapper around ``psycopg2.pool.ThreadedConnectionPool``.

    Not ``SimpleConnectionPool``: this process hands out the pool to
    concurrent callers on more than one thread - FastAPI's sync (non-
    ``async def``) routes each run in the AnyIO threadpool, and
    ``app.api.rag_ops_sync.RagOpsConfigPoller`` additionally polls
    ``RagOpsRepository.get_config()`` from its own background thread via
    ``asyncio.to_thread``. ``SimpleConnectionPool``'s ``getconn``/``putconn``
    aren't safe under concurrent access from multiple threads (only
    ``ThreadedConnectionPool`` locks around them), so two callers racing
    for a connection could corrupt the pool's internal bookkeeping.
    """

    def __init__(self, dsn: str, minconn: int = 1, maxconn: int = 10) -> None:
        self._pool: pg_pool.ThreadedConnectionPool = pg_pool.ThreadedConnectionPool(
            minconn, maxconn, dsn
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
