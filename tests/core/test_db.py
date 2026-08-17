from __future__ import annotations

import app.core.db as db


class _FakePgPool:
    def __init__(self, minconn: int, maxconn: int, dsn: str) -> None:
        self.minconn = minconn
        self.maxconn = maxconn
        self.dsn = dsn
        self.getconn_calls = 0
        self.putconn_calls: list[object] = []
        self.closeall_calls = 0

    def getconn(self) -> str:
        self.getconn_calls += 1
        return "connection"

    def putconn(self, conn: object) -> None:
        self.putconn_calls.append(conn)

    def closeall(self) -> None:
        self.closeall_calls += 1


def test_postgres_connection_pool_uses_threaded_connection_pool_not_simple(
    monkeypatch,
) -> None:
    """Regression: psycopg2.pool.SimpleConnectionPool's getconn()/putconn()
    aren't safe under concurrent access from multiple threads - this app's
    sync (non-``async def``) FastAPI routes run in AnyIO's threadpool, and
    ``app.api.rag_ops_sync.RagOpsConfigPoller`` additionally polls from its
    own background thread via ``asyncio.to_thread``, sharing this same
    process-level pool. The underlying pool class must be the thread-safe
    ``ThreadedConnectionPool``, not ``SimpleConnectionPool``."""
    constructed: dict[str, object] = {}

    class _RecordingThreadedPool(_FakePgPool):
        def __init__(self, minconn: int, maxconn: int, dsn: str) -> None:
            constructed["called_with"] = (minconn, maxconn, dsn)
            super().__init__(minconn, maxconn, dsn)

    def _must_not_be_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("PostgresConnectionPool must not construct a SimpleConnectionPool")

    monkeypatch.setattr(db.pg_pool, "ThreadedConnectionPool", _RecordingThreadedPool)
    monkeypatch.setattr(db.pg_pool, "SimpleConnectionPool", _must_not_be_called)

    pool = db.PostgresConnectionPool("postgresql://user:pass@localhost/db", minconn=2, maxconn=5)

    assert constructed["called_with"] == (2, 5, "postgresql://user:pass@localhost/db")
    assert isinstance(pool._pool, _RecordingThreadedPool)


def test_connection_checks_out_and_returns_a_connection(monkeypatch) -> None:
    monkeypatch.setattr(db.pg_pool, "ThreadedConnectionPool", _FakePgPool)

    pool = db.PostgresConnectionPool("postgresql://user:pass@localhost/db")
    fake_pool: _FakePgPool = pool._pool  # type: ignore[assignment]

    with pool.connection() as conn:
        assert conn == "connection"
        assert fake_pool.getconn_calls == 1
        assert fake_pool.putconn_calls == []

    assert fake_pool.putconn_calls == ["connection"]


def test_connection_returns_the_connection_even_if_the_caller_raises(monkeypatch) -> None:
    monkeypatch.setattr(db.pg_pool, "ThreadedConnectionPool", _FakePgPool)

    pool = db.PostgresConnectionPool("postgresql://user:pass@localhost/db")
    fake_pool: _FakePgPool = pool._pool  # type: ignore[assignment]

    try:
        with pool.connection():
            raise ValueError("boom")
    except ValueError:
        pass

    assert fake_pool.putconn_calls == ["connection"]


def test_close_closes_every_pooled_connection(monkeypatch) -> None:
    monkeypatch.setattr(db.pg_pool, "ThreadedConnectionPool", _FakePgPool)

    pool = db.PostgresConnectionPool("postgresql://user:pass@localhost/db")
    fake_pool: _FakePgPool = pool._pool  # type: ignore[assignment]

    pool.close()

    assert fake_pool.closeall_calls == 1
