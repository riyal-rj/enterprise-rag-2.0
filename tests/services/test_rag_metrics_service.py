from __future__ import annotations

from app.services.rag_metrics_service import RagMetricsService


def test_rerank_stats_empty_by_default() -> None:
    metrics = RagMetricsService()

    stats = metrics.rerank_stats()

    assert stats.sample_count == 0
    assert stats.p50_latency_ms is None
    assert stats.p95_latency_ms is None
    assert stats.fallback_rate == 0.0
    assert stats.voyage_tokens_total == 0


def test_rerank_stats_computes_percentiles_and_fallback_rate() -> None:
    metrics = RagMetricsService()

    for duration_ms in [10.0, 20.0, 30.0, 40.0]:
        metrics.record_rerank(duration_ms=duration_ms, fallback=False, usage_tokens=None)
    metrics.record_rerank(duration_ms=1000.0, fallback=True, usage_tokens=None)

    stats = metrics.rerank_stats()

    assert stats.sample_count == 5
    assert stats.fallback_rate == 0.2
    assert stats.p50_latency_ms is not None
    assert stats.p95_latency_ms is not None


def test_rerank_stats_accumulates_voyage_tokens_across_samples() -> None:
    metrics = RagMetricsService()

    metrics.record_rerank(duration_ms=5.0, fallback=False, usage_tokens=100)
    metrics.record_rerank(duration_ms=5.0, fallback=False, usage_tokens=50)
    metrics.record_rerank(duration_ms=5.0, fallback=False, usage_tokens=None)

    assert metrics.rerank_stats().voyage_tokens_total == 150


def test_rerank_stats_respects_rolling_window() -> None:
    metrics = RagMetricsService(window_size=3)

    for _ in range(2):
        metrics.record_rerank(duration_ms=1.0, fallback=True, usage_tokens=None)
    for _ in range(3):
        metrics.record_rerank(duration_ms=1.0, fallback=False, usage_tokens=None)

    stats = metrics.rerank_stats()

    # Only the most recent 3 samples survive - all non-fallback.
    assert stats.sample_count == 3
    assert stats.fallback_rate == 0.0


def test_semantic_cache_stats_empty_by_default() -> None:
    metrics = RagMetricsService()

    stats = metrics.semantic_cache_stats()

    assert stats.lookups == 0
    assert stats.hits == 0
    assert stats.hit_rate == 0.0


def test_semantic_cache_stats_tracks_hit_rate() -> None:
    metrics = RagMetricsService()

    metrics.record_semantic_cache_lookup(hit=True)
    metrics.record_semantic_cache_lookup(hit=True)
    metrics.record_semantic_cache_lookup(hit=False)

    stats = metrics.semantic_cache_stats()

    assert stats.lookups == 3
    assert stats.hits == 2
    assert stats.hit_rate == 2 / 3
