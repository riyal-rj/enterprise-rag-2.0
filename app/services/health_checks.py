"""Dependency health checks (Strategy pattern) and their aggregator.

Each external dependency (Postgres, Qdrant, Redis, OpenAI, Tavily) gets its
own small ``HealthCheck``. Adding a new one to the ``/admin/health`` report
means adding one class and one line in ``app.api.deps`` — nothing else
changes. Every check deliberately swallows its own exceptions: a health
probe must never raise, since one dependency being down shouldn't crash the
health endpoint itself.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Protocol

from app.core.db import PostgresConnectionPool
from app.services.web_search import search_web

if TYPE_CHECKING:
    from upstash_redis import Redis

logger = logging.getLogger(__name__)


class HealthCheck(Protocol):
    """A single dependency probe."""

    name: str

    async def check(self) -> bool: ...


class PostgresHealthCheck:
    name = "postgres"

    def __init__(self, pool: PostgresConnectionPool) -> None:
        self._pool = pool

    async def check(self) -> bool:
        return await asyncio.to_thread(self._check_sync)

    def _check_sync(self) -> bool:
        try:
            with self._pool.connection() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception as exc:  # best-effort probe: never propagate
            logger.debug("Postgres health check failed: %s", exc)
            return False


class QdrantHealthCheck:
    name = "qdrant"

    def __init__(self, url: str, timeout_seconds: float = 2.0) -> None:
        self._url = url
        self._timeout_seconds = timeout_seconds

    async def check(self) -> bool:
        return await asyncio.to_thread(self._check_sync)

    def _check_sync(self) -> bool:
        try:
            from qdrant_client import QdrantClient

            QdrantClient(url=self._url, timeout=self._timeout_seconds).get_collections()
            return True
        except Exception as exc:  # best-effort probe: never propagate
            logger.debug("Qdrant health check failed: %s", exc)
            return False


class RedisHealthCheck:
    name = "redis"

    def __init__(self, redis_client: Redis) -> None:
        self._redis = redis_client

    async def check(self) -> bool:
        return await asyncio.to_thread(self._check_sync)

    def _check_sync(self) -> bool:
        try:
            self._redis.ping()
            return True
        except Exception as exc:  # best-effort probe: never propagate
            logger.debug("Redis health check failed: %s", exc)
            return False


class OpenAIHealthCheck:
    name = "openai"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def check(self) -> bool:
        try:
            from openai import AsyncOpenAI

            await AsyncOpenAI(api_key=self._api_key).models.list()
            return True
        except Exception as exc:  # best-effort probe: never propagate
            logger.debug("OpenAI health check failed: %s", exc)
            return False


class TavilyHealthCheck:
    name = "tavily"

    async def check(self) -> bool:
        try:
            await asyncio.to_thread(search_web, "health check")
            return True
        except ValueError:
            # Not configured — treated as healthy, same as an optional
            # feature that simply isn't enabled.
            return True
        except Exception as exc:  # best-effort probe: never propagate
            logger.debug("Tavily health check failed: %s", exc)
            return False


class HealthCheckService:
    """Runs every registered :class:`HealthCheck` concurrently."""

    def __init__(self, checks: list[HealthCheck]) -> None:
        self._checks = checks

    async def run(self) -> dict[str, bool]:
        results = await asyncio.gather(
            *(check.check() for check in self._checks), return_exceptions=True
        )
        return {
            check.name: (result if isinstance(result, bool) else False)
            for check, result in zip(self._checks, results, strict=True)
        }
