"""In-memory, process-local performance metrics for the RAG Operations panel.

Deliberately not persisted - a restart resets it, same as
``QueryCacheService``'s per-tier hit/miss/set counters. Only *config*
(``app.repositories.rag_ops_repository``) needs to survive a restart;
point-in-time latency/hit-rate samples don't, and persisting them would
mean a write on every single chat request just to power a dashboard.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

_DEFAULT_WINDOW_SIZE = 1000


@dataclass(frozen=True)
class RerankMetricsSnapshot:
    """Reranker performance over the last ``window_size`` *attempted*
    reranks (backend actually invoked - rollout-bypassed and
    emergency-disabled requests don't count as attempts). ``None``
    latencies mean no rerank has been attempted yet since the process
    started."""

    sample_count: int
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    fallback_rate: float
    voyage_tokens_total: int


@dataclass(frozen=True)
class SemanticCacheMetricsSnapshot:
    lookups: int
    hits: int
    hit_rate: float


@dataclass(frozen=True)
class HyDEMetricsSnapshot:
    """HyDE performance over the last ``window_size`` *attempted* transforms
    (the delegate was actually invoked - rollout-bypassed and
    emergency-disabled requests don't count as attempts, same distinction as
    :class:`RerankMetricsSnapshot`)."""

    sample_count: int
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    fallback_rate: float
    usage_tokens_total: int
    rollout_bypasses: int
    emergency_bypasses: int


class RagMetricsService:
    """Records reranker/HyDE latency/fallback/token-usage samples and
    semantic cache lookup outcomes; exposes rolled-up snapshots for the RAG
    Operations panel's status endpoint."""

    def __init__(self, window_size: int = _DEFAULT_WINDOW_SIZE) -> None:
        self._lock = threading.Lock()
        # (duration_ms, fallback) kept together so p50/p95/fallback_rate
        # all describe the exact same rolling window, rather than latency
        # being windowed while fallback count drifts out of sync with it.
        self._rerank_samples: deque[tuple[float, bool]] = deque(maxlen=window_size)
        # Cumulative, not windowed: this is a cost metric (Voyage bills per
        # token), so admins need the running total, not an estimate that
        # shrinks as older samples fall out of the window.
        self._voyage_tokens_total = 0
        self._semantic_lookups = 0
        self._semantic_hits = 0
        self._hyde_samples: deque[tuple[float, bool]] = deque(maxlen=window_size)
        self._hyde_usage_tokens_total = 0
        self._hyde_bypass_counts: dict[str, int] = {"rollout": 0, "emergency_disabled": 0}

    def record_rerank(self, *, duration_ms: float, fallback: bool, usage_tokens: int | None) -> None:
        with self._lock:
            self._rerank_samples.append((duration_ms, fallback))
            if usage_tokens:
                self._voyage_tokens_total += usage_tokens

    def rerank_stats(self) -> RerankMetricsSnapshot:
        with self._lock:
            samples = list(self._rerank_samples)
            voyage_tokens_total = self._voyage_tokens_total

        durations = sorted(duration for duration, _ in samples)
        attempts = len(samples)
        fallback_count = sum(1 for _, fallback in samples if fallback)
        return RerankMetricsSnapshot(
            sample_count=attempts,
            p50_latency_ms=_percentile(durations, 0.50),
            p95_latency_ms=_percentile(durations, 0.95),
            fallback_rate=(fallback_count / attempts) if attempts else 0.0,
            voyage_tokens_total=voyage_tokens_total,
        )

    def record_semantic_cache_lookup(self, *, hit: bool) -> None:
        with self._lock:
            self._semantic_lookups += 1
            if hit:
                self._semantic_hits += 1

    def semantic_cache_stats(self) -> SemanticCacheMetricsSnapshot:
        with self._lock:
            lookups = self._semantic_lookups
            hits = self._semantic_hits
        return SemanticCacheMetricsSnapshot(
            lookups=lookups, hits=hits, hit_rate=(hits / lookups) if lookups else 0.0
        )

    def record_hyde_attempt(self, *, duration_ms: float, fallback: bool, usage_tokens: int) -> None:
        with self._lock:
            self._hyde_samples.append((duration_ms, fallback))
            self._hyde_usage_tokens_total += max(usage_tokens, 0)

    def record_hyde_bypass(self, *, reason: str) -> None:
        with self._lock:
            self._hyde_bypass_counts[reason] = self._hyde_bypass_counts.get(reason, 0) + 1

    def hyde_stats(self) -> HyDEMetricsSnapshot:
        with self._lock:
            samples = list(self._hyde_samples)
            tokens = self._hyde_usage_tokens_total
            bypasses = dict(self._hyde_bypass_counts)

        durations = sorted(duration for duration, _ in samples)
        attempts = len(samples)
        fallback_count = sum(1 for _, fallback in samples if fallback)
        return HyDEMetricsSnapshot(
            sample_count=attempts,
            p50_latency_ms=_percentile(durations, 0.50),
            p95_latency_ms=_percentile(durations, 0.95),
            fallback_rate=(fallback_count / attempts) if attempts else 0.0,
            usage_tokens_total=tokens,
            rollout_bypasses=bypasses.get("rollout", 0),
            emergency_bypasses=bypasses.get("emergency_disabled", 0),
        )


def _percentile(sorted_samples: list[float], fraction: float) -> float | None:
    if not sorted_samples:
        return None
    index = min(len(sorted_samples) - 1, int(len(sorted_samples) * fraction))
    return sorted_samples[index]
