from __future__ import annotations

from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.dynamic_reranker import DynamicReranker
from app.rag_services.rag_runtime_config import RagRuntimeConfig, RagRuntimeConfigStore
from app.rag_services.reranker import ReRankedChunk, ReRankOutcome
from app.services.rag_metrics_service import RagMetricsService


class _FakeReranker:
    def __init__(self, name: str = "fake", raise_error: bool = False) -> None:
        self._name = name
        self._raise_error = raise_error
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def cache_namespace(self) -> str:
        return f"reranker:{self._name}:v1"

    def rerank(self, *, query, candidates, top_k):
        self.calls += 1
        if self._raise_error:
            raise RuntimeError("boom")
        items = tuple(
            ReRankedChunk(chunk=c, original_rank=i + 1, rerank_score=1.0)
            for i, c in enumerate(candidates[:top_k])
        )
        return ReRankOutcome(items=items, backend=self._name, applied=True)


def _candidates() -> list[RetrievedChunk]:
    return [RetrievedChunk(text="a", source="doc.pdf", score=0.9, page_number=1)]


def _config_store(
    *,
    backend: str = "local",
    rollout_percentage: int = 100,
    emergency_disabled: bool = False,
) -> RagRuntimeConfigStore:
    return RagRuntimeConfigStore(
        RagRuntimeConfig(
            reranking_enabled=True,
            reranker_backend=backend,  # type: ignore[arg-type]
            reranker_rollout_percentage=rollout_percentage,
            reranker_emergency_disabled=emergency_disabled,
            semantic_cache_enabled=False,
            semantic_cache_threshold=0.95,
            corpus_version=1,
        )
    )


def test_rerank_uses_local_backend_by_default() -> None:
    local = _FakeReranker("local")
    voyage = _FakeReranker("voyage")
    reranker = DynamicReranker(
        local=local, voyage=voyage, metrics=RagMetricsService(), config_store=_config_store()
    )

    outcome = reranker.rerank(query="q", candidates=_candidates(), top_k=1)

    assert outcome.backend == "local"
    assert local.calls == 1
    assert voyage.calls == 0


def test_rerank_switches_backend_after_the_store_replaces_the_snapshot() -> None:
    local = _FakeReranker("local")
    voyage = _FakeReranker("voyage")
    store = _config_store(backend="local")
    reranker = DynamicReranker(
        local=local, voyage=voyage, metrics=RagMetricsService(), config_store=store
    )

    store.replace(
        RagRuntimeConfig(
            reranking_enabled=True,
            reranker_backend="voyage",
            reranker_rollout_percentage=100,
            reranker_emergency_disabled=False,
            semantic_cache_enabled=False,
            semantic_cache_threshold=0.95,
            corpus_version=1,
        )
    )
    outcome = reranker.rerank(query="q", candidates=_candidates(), top_k=1)

    assert outcome.backend == "voyage"
    assert voyage.calls == 1


def test_emergency_disabled_bypasses_the_backend_entirely() -> None:
    local = _FakeReranker("local")
    reranker = DynamicReranker(
        local=local,
        voyage=None,
        metrics=RagMetricsService(),
        config_store=_config_store(emergency_disabled=True),
    )

    outcome = reranker.rerank(query="q", candidates=_candidates(), top_k=1)

    assert outcome.applied is False
    assert outcome.backend == "none"
    assert local.calls == 0


def test_rollout_zero_percent_bypasses_the_backend() -> None:
    local = _FakeReranker("local")
    reranker = DynamicReranker(
        local=local,
        voyage=None,
        metrics=RagMetricsService(),
        config_store=_config_store(rollout_percentage=0),
    )

    outcome = reranker.rerank(query="any question", candidates=_candidates(), top_k=1)

    assert outcome.applied is False
    assert local.calls == 0


def test_rollout_hundred_percent_always_attempts_the_backend() -> None:
    local = _FakeReranker("local")
    reranker = DynamicReranker(
        local=local,
        voyage=None,
        metrics=RagMetricsService(),
        config_store=_config_store(rollout_percentage=100),
    )

    for question in ["a", "b", "c", "d", "e"]:
        reranker.rerank(query=question, candidates=_candidates(), top_k=1)

    assert local.calls == 5


def test_rollout_sampling_is_deterministic_per_question() -> None:
    local = _FakeReranker("local")
    reranker = DynamicReranker(
        local=local,
        voyage=None,
        metrics=RagMetricsService(),
        config_store=_config_store(rollout_percentage=50),
    )

    first_run = [
        reranker.rerank(query=f"question {i}", candidates=_candidates(), top_k=1).applied
        for i in range(20)
    ]
    second_run = [
        reranker.rerank(query=f"question {i}", candidates=_candidates(), top_k=1).applied
        for i in range(20)
    ]

    assert first_run == second_run


def test_voyage_backend_without_delegate_fails_open_to_noop() -> None:
    local = _FakeReranker("local")
    reranker = DynamicReranker(
        local=local,
        voyage=None,
        metrics=RagMetricsService(),
        config_store=_config_store(backend="voyage"),
    )

    outcome = reranker.rerank(query="q", candidates=_candidates(), top_k=1)

    assert outcome.applied is False
    assert outcome.fallback is True
    assert local.calls == 0


def test_backend_failure_falls_back_and_still_records_metrics() -> None:
    local = _FakeReranker("local", raise_error=True)
    metrics = RagMetricsService()
    reranker = DynamicReranker(
        local=local, voyage=None, metrics=metrics, config_store=_config_store()
    )

    outcome = reranker.rerank(query="q", candidates=_candidates(), top_k=1)

    assert outcome.applied is False
    assert outcome.fallback is True
    stats = metrics.rerank_stats()
    assert stats.sample_count == 1
    assert stats.fallback_rate == 1.0


def test_has_voyage_backend_reflects_configured_delegate() -> None:
    with_voyage = DynamicReranker(
        local=_FakeReranker(),
        voyage=_FakeReranker("voyage"),
        metrics=RagMetricsService(),
        config_store=_config_store(),
    )
    without_voyage = DynamicReranker(
        local=_FakeReranker(),
        voyage=None,
        metrics=RagMetricsService(),
        config_store=_config_store(),
    )

    assert with_voyage.has_voyage_backend is True
    assert without_voyage.has_voyage_backend is False


def test_cache_namespace_changes_when_backend_or_rollout_changes() -> None:
    store = _config_store()
    reranker = DynamicReranker(
        local=_FakeReranker("local"),
        voyage=_FakeReranker("voyage"),
        metrics=RagMetricsService(),
        config_store=store,
    )

    baseline = reranker.cache_namespace
    store.replace(
        RagRuntimeConfig(
            reranking_enabled=True,
            reranker_backend="local",
            reranker_rollout_percentage=50,
            reranker_emergency_disabled=False,
            semantic_cache_enabled=False,
            semantic_cache_threshold=0.95,
            corpus_version=1,
        )
    )
    after_rollout_change = reranker.cache_namespace
    store.replace(
        RagRuntimeConfig(
            reranking_enabled=True,
            reranker_backend="voyage",
            reranker_rollout_percentage=50,
            reranker_emergency_disabled=False,
            semantic_cache_enabled=False,
            semantic_cache_threshold=0.95,
            corpus_version=1,
        )
    )
    after_backend_change = reranker.cache_namespace

    assert baseline != after_rollout_change
    assert after_rollout_change != after_backend_change


def test_config_replacement_is_a_single_atomic_swap_never_a_torn_mix() -> None:
    """Regression: apply_rag_ops_config() used to call reranker.set_backend()/
    set_rollout_percentage()/set_emergency_disabled() as three separate,
    non-atomic mutations - a concurrent rerank() call landing between them
    could see e.g. the new backend with the old emergency-disabled state.
    Replacing the whole snapshot in one assignment makes that impossible:
    every rerank() call sees either the fully-old or fully-new config."""
    local = _FakeReranker("local")
    voyage = _FakeReranker("voyage")
    store = _config_store(backend="local", emergency_disabled=False)
    reranker = DynamicReranker(
        local=local, voyage=voyage, metrics=RagMetricsService(), config_store=store
    )

    # Snapshot captured by one rerank() call must be internally consistent
    # even if the store is replaced again immediately after.
    outcome_before = reranker.rerank(query="q", candidates=_candidates(), top_k=1)
    store.replace(
        RagRuntimeConfig(
            reranking_enabled=True,
            reranker_backend="voyage",
            reranker_rollout_percentage=100,
            reranker_emergency_disabled=True,
            semantic_cache_enabled=False,
            semantic_cache_threshold=0.95,
            corpus_version=1,
        )
    )
    outcome_after = reranker.rerank(query="q", candidates=_candidates(), top_k=1)

    assert outcome_before.backend == "local"
    assert outcome_before.applied is True
    # New config is backend=voyage *and* emergency_disabled=True together -
    # never voyage-with-emergency-off or local-with-emergency-on.
    assert outcome_after.applied is False
    assert outcome_after.backend == "none"
    assert voyage.calls == 0
