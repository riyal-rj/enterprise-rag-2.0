"""Tiered query cache with hit/miss/set counters.

``QueryCacheService`` is the shared cache other pipeline stages (embeddings,
RAG answers, SQL generation/results, intent routing) read and write through;
this module only defines the cache itself and its admin-facing stats/clear
operations. ``CacheBackend`` is an Adapter over the actual store (Upstash
Redis here), so the backend can change without touching call sites.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol, TypedDict

from app.core.config.cache import CacheSettings

if TYPE_CHECKING:
    from upstash_redis import Redis


class CacheTier(str, Enum):
    """Named cache partitions, one per TTL in :class:`CacheSettings`."""

    EMBEDDING = "embedding"
    RAG_ANSWER = "rag_answer"
    SQL_GEN = "sql_gen"
    SQL_RESULT = "sql_result"
    INTENT = "intent"


class CacheBackend(Protocol):
    """Key/value store contract the cache service reads and writes through."""

    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str, ttl_seconds: int) -> None: ...

    def delete_prefix(self, prefix: str) -> int: ...


class UpstashCacheBackend:
    """:class:`CacheBackend` backed by Upstash Redis's REST client."""

    def __init__(self, redis_client: Redis) -> None:
        self._redis = redis_client

    def get(self, key: str) -> str | None:
        return self._redis.get(key)

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self._redis.set(key, value, ex=ttl_seconds)

    def delete_prefix(self, prefix: str) -> int:
        keys = self._redis.keys(f"{prefix}*")
        if not keys:
            return 0
        self._redis.delete(*keys)
        return len(keys)


class TierStatsDict(TypedDict):
    hits: int
    misses: int
    sets: int
    hit_rate: float


@dataclass
class _TierCounters:
    hits: int = 0
    misses: int = 0
    sets: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0

    def as_dict(self) -> TierStatsDict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "hit_rate": self.hit_rate,
        }


_TTL_BY_TIER_FIELD: dict[CacheTier, str] = {
    CacheTier.EMBEDDING: "cache_ttl_embeddings",
    CacheTier.RAG_ANSWER: "cache_ttl_rag",
    CacheTier.SQL_GEN: "cache_ttl_sql_gen",
    CacheTier.SQL_RESULT: "cache_ttl_sql_result",
    CacheTier.INTENT: "cache_ttl_intent",
}


class QueryCacheService:
    """Per-tier cache reads/writes plus admin-facing stats and clearing."""

    def __init__(self, backend: CacheBackend, ttls: CacheSettings) -> None:
        self._backend = backend
        self._ttls = ttls
        self._lock = threading.Lock()
        self._counters: dict[CacheTier, _TierCounters] = {
            tier: _TierCounters() for tier in CacheTier
        }

    def get(self, tier: CacheTier, key: str) -> str | None:
        value = self._backend.get(self._namespaced_key(tier, key))
        with self._lock:
            if value is None:
                self._counters[tier].misses += 1
            else:
                self._counters[tier].hits += 1
        return value

    def set(self, tier: CacheTier, key: str, value: str) -> None:
        self._backend.set(self._namespaced_key(tier, key), value, self._ttl_for(tier))
        with self._lock:
            self._counters[tier].sets += 1

    def stats(self) -> dict[str, TierStatsDict]:
        with self._lock:
            return {tier.value: counters.as_dict() for tier, counters in self._counters.items()}

    def clear(self) -> dict[str, int]:
        """Clear every tier's stored entries and reset its counters."""
        cleared: dict[str, int] = {}
        with self._lock:
            for tier in CacheTier:
                cleared[tier.value] = self._backend.delete_prefix(f"{tier.value}:")
                self._counters[tier] = _TierCounters()
        return cleared

    def _namespaced_key(self, 
                        tier: CacheTier, 
                        key: str) -> str:
        return f"{tier.value}:{key}"

    def _ttl_for(self, tier: CacheTier) -> int:
        return getattr(self._ttls, _TTL_BY_TIER_FIELD[tier])
