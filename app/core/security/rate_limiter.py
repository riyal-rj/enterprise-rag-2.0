"""Request rate limiting, behind a swappable interface (Strategy/Adapter).

``InMemorySlidingWindowRateLimiter`` is correct for a single process only
(handy for local dev/tests). ``UpstashSlidingWindowRateLimiter`` shares
limits across every app instance via a Redis sorted set and is the one
wired up in production (see ``app.api.deps.get_rate_limiter``).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from upstash_redis import Redis


class RateLimiter(Protocol):
    """Rate-limiting contract.

    Returns ``(allowed, remaining, reset_at)``, where ``reset_at`` is a
    timestamp for when the window next has room (clock base is
    implementation-defined: monotonic for the in-memory limiter, wall-clock
    for Redis-backed ones — callers should treat it as opaque and only use
    it as a hint, e.g. for a ``Retry-After`` header).
    """

    def is_allowed(
        self, 
        key: str, 
        route: str, 
        limit: int, 
        window_seconds: int
    ) -> tuple[bool, int, float]: ...


class InMemorySlidingWindowRateLimiter:
    """Thread-safe, per-process sliding-window rate limiter."""

    def __init__(self) -> None:
        self._hits: dict[tuple[str, str], deque[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(
        self, 
        key: str, 
        route: str, 
        limit: int, 
        window_seconds: int
    ) -> tuple[bool, int, float]:
        now = time.monotonic()
        bucket_key = (key, route)
        with self._lock:
            hits = self._hits.setdefault(bucket_key, deque())
            cutoff = now - window_seconds
            while hits and hits[0] < cutoff:
                hits.popleft()

            if len(hits) >= limit:
                return False, 0, hits[0] + window_seconds

            hits.append(now)
            return True, limit - len(hits), hits[0] + window_seconds


class UpstashSlidingWindowRateLimiter:
    """Redis-backed sliding-window rate limiter (sorted-set algorithm).

    Shares limits across every process/instance hitting the same Upstash
    Redis database, unlike the in-memory limiter.
    """

    def __init__(self, redis_client: Redis) -> None:
        self._redis = redis_client

    def is_allowed(
        self, key: str, route: str, limit: int, window_seconds: int
    ) -> tuple[bool, int, float]:
        redis_key = f"rate_limit:{key}:{route}"
        now = time.time()
        window_start = now - window_seconds

        pipe = self._redis.pipeline()
        pipe.zremrangebyscore(redis_key, 0, window_start)
        pipe.zadd(redis_key, {str(now): now})
        pipe.zcard(redis_key)
        pipe.zrange(redis_key, 0, 0, withscores=True)
        pipe.expire(redis_key, window_seconds)
        results = pipe.exec()

        request_count: int = cast(int, results[2])
        oldest_entry = cast(list[tuple[str, float]], results[3]) or []
        oldest_score = oldest_entry[0][1] if oldest_entry else now

        allowed = request_count <= limit
        remaining = max(0, limit - request_count)
        reset_at = oldest_score + window_seconds
        return allowed, remaining, reset_at
