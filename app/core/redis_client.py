"""Shared Upstash Redis client construction.

A single client is built once (see ``app.api.deps.get_redis_client``) and
injected into every Redis-backed component (rate limiter, query cache,
health check), instead of each one opening its own connection.
"""

from __future__ import annotations

from upstash_redis import Redis


def build_redis_client(url: str, token: str) -> Redis:
    return Redis(url=url, token=token)
