"""Response schemas for ``/admin/cache/*``."""

from __future__ import annotations

from pydantic import BaseModel


class CacheTierStats(BaseModel):
    hits: int
    misses: int
    sets: int
    hit_rate: float


class CacheStatsResponse(BaseModel):
    embedding: CacheTierStats
    rag: CacheTierStats
    sql_gen: CacheTierStats
    sql_result: CacheTierStats
    intent_router: CacheTierStats


class CacheClearResponse(BaseModel):
    status: str
    cleared: dict[str, int]
