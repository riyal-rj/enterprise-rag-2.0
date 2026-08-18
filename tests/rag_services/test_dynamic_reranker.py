from __future__ import annotations

from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.dynamic_reranker import DynamicReranker
from app.rag_services.rag_runtime_config import RagRuntimeConfig
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


def _config(
    *,
    backend: str = "local",
    rollout_percentage: int = 100,
    emergency_disabled: bool = False,
) -> RagRuntimeConfig:
    return RagRuntimeConfig(
        reranking_enabled=True,
        reranker_backend=backend,  # type: ignore[arg-type]
        reranker_rollout_percentage=rollout_percentage,
        emergency_disabled=emergency_disabled,
        semantic_cache_enabled=False,
        semantic_cache_threshold=0.95,
        corpus_version=1,
        hyde_enabled=False,
        hyde_rollout_percentage=0,
        crag_enabled=False,
        crag_rollout_percentage=0,
        crag_web_enabled=False,
    )


def test_plan_is_disabled_cohort_when_not_enabled() -> None:
    reranker = DynamicReranker(local=_FakeReranker(), voyage=None, metrics=RagMetricsService())

    plan = reranker.plan("q", _config(), enabled=False)

    assert plan.cohort == "disabled"
    assert plan.bypass_reason == "disabled"


def test_plan_uses_local_backend_by_default() -> None:
    local = _FakeReranker("local")
    voyage = _FakeReranker("voyage")
    reranker = DynamicReranker(local=local, voyage=voyage, metrics=RagMetricsService())

    plan = reranker.plan("q", _config(backend="local"), enabled=True)
    outcome = reranker.execute(plan, query="q", candidates=_candidates(), top_k=1)

    assert plan.cohort == "treatment"
    assert plan.backend_key == "local"
    assert outcome.backend == "local"
    assert local.calls == 1
    assert voyage.calls == 0


def test_plan_selects_voyage_backend_when_configured() -> None:
    local = _FakeReranker("local")
    voyage = _FakeReranker("voyage")
    reranker = DynamicReranker(local=local, voyage=voyage, metrics=RagMetricsService())

    plan = reranker.plan("q", _config(backend="voyage"), enabled=True)
    outcome = reranker.execute(plan, query="q", candidates=_candidates(), top_k=1)

    assert plan.backend_key == "voyage"
    assert outcome.backend == "voyage"
    assert voyage.calls == 1
    assert local.calls == 0


def test_emergency_disabled_plan_is_disabled_cohort_and_execute_is_noop() -> None:
    local = _FakeReranker("local")
    reranker = DynamicReranker(local=local, voyage=None, metrics=RagMetricsService())

    plan = reranker.plan("q", _config(emergency_disabled=True), enabled=True)
    outcome = reranker.execute(plan, query="q", candidates=_candidates(), top_k=1)

    assert plan.cohort == "disabled"
    assert plan.bypass_reason == "emergency_disabled"
    assert outcome.applied is False
    assert outcome.backend == "none"
    assert local.calls == 0


def test_rollout_zero_percent_is_control_cohort() -> None:
    local = _FakeReranker("local")
    reranker = DynamicReranker(local=local, voyage=None, metrics=RagMetricsService())

    plan = reranker.plan("any question", _config(rollout_percentage=0), enabled=True)
    outcome = reranker.execute(plan, query="any question", candidates=_candidates(), top_k=1)

    assert plan.cohort == "control"
    assert plan.bypass_reason == "rollout"
    assert outcome.applied is False
    assert local.calls == 0


def test_rollout_hundred_percent_always_yields_treatment() -> None:
    local = _FakeReranker("local")
    reranker = DynamicReranker(local=local, voyage=None, metrics=RagMetricsService())
    config = _config(rollout_percentage=100)

    for question in ["a", "b", "c", "d", "e"]:
        plan = reranker.plan(question, config, enabled=True)
        reranker.execute(plan, query=question, candidates=_candidates(), top_k=1)

    assert local.calls == 5


def test_rollout_sampling_is_deterministic_per_question() -> None:
    reranker = DynamicReranker(local=_FakeReranker(), voyage=None, metrics=RagMetricsService())
    config = _config(rollout_percentage=50)

    first_run = [reranker.plan(f"question {i}", config, enabled=True).cohort for i in range(20)]
    second_run = [reranker.plan(f"question {i}", config, enabled=True).cohort for i in range(20)]

    assert first_run == second_run


def test_voyage_backend_without_delegate_fails_open_to_noop() -> None:
    local = _FakeReranker("local")
    reranker = DynamicReranker(local=local, voyage=None, metrics=RagMetricsService())

    plan = reranker.plan("q", _config(backend="voyage"), enabled=True)
    outcome = reranker.execute(plan, query="q", candidates=_candidates(), top_k=1)

    assert plan.cohort == "treatment"
    assert outcome.applied is False
    assert outcome.fallback is True
    assert local.calls == 0


def test_backend_failure_falls_back_and_still_records_metrics() -> None:
    local = _FakeReranker("local", raise_error=True)
    metrics = RagMetricsService()
    reranker = DynamicReranker(local=local, voyage=None, metrics=metrics)

    plan = reranker.plan("q", _config(), enabled=True)
    outcome = reranker.execute(plan, query="q", candidates=_candidates(), top_k=1)

    assert outcome.applied is False
    assert outcome.fallback is True
    stats = metrics.rerank_stats()
    assert stats.sample_count == 1
    assert stats.fallback_rate == 1.0


def test_has_voyage_backend_reflects_configured_delegate() -> None:
    with_voyage = DynamicReranker(
        local=_FakeReranker(), voyage=_FakeReranker("voyage"), metrics=RagMetricsService()
    )
    without_voyage = DynamicReranker(
        local=_FakeReranker(), voyage=None, metrics=RagMetricsService()
    )

    assert with_voyage.has_voyage_backend is True
    assert without_voyage.has_voyage_backend is False


def test_cache_namespace_changes_when_backend_or_rollout_changes() -> None:
    reranker = DynamicReranker(
        local=_FakeReranker("local"), voyage=_FakeReranker("voyage"), metrics=RagMetricsService()
    )

    baseline = reranker.plan("q", _config(backend="local", rollout_percentage=100), enabled=True)
    after_backend_change = reranker.plan(
        "q", _config(backend="voyage", rollout_percentage=100), enabled=True
    )

    assert baseline.cache_namespace != after_backend_change.cache_namespace


def test_cohort_isolation_a_control_and_treatment_query_never_share_a_cache_namespace() -> None:
    """Regression: DynamicReranker.cache_namespace used to only encode the
    *configured* rollout percentage (e.g. "rollout=30"), not which cohort a
    specific query actually landed in - so a control-cohort query and a
    treatment-cohort query under the same 30% rollout would share one cache
    namespace, letting a semantic-cache hit serve one query an answer
    generated under the other's (different) reranking decision. plan()'s
    cache_namespace must instead encode the *resolved* cohort."""
    reranker = DynamicReranker(
        local=_FakeReranker("local"), voyage=None, metrics=RagMetricsService()
    )
    config = _config(rollout_percentage=50)

    # Find one query that samples into each cohort under this config.
    control_query = next(
        q
        for q in (f"q{i}" for i in range(50))
        if reranker.plan(q, config, enabled=True).cohort == "control"
    )
    treatment_query = next(
        q
        for q in (f"q{i}" for i in range(50))
        if reranker.plan(q, config, enabled=True).cohort == "treatment"
    )

    control_plan = reranker.plan(control_query, config, enabled=True)
    treatment_plan = reranker.plan(treatment_query, config, enabled=True)

    assert control_plan.cache_namespace != treatment_plan.cache_namespace


def test_execute_is_driven_entirely_by_the_plan_not_by_a_later_config_change() -> None:
    """DynamicReranker holds no config/store reference of its own - plan()
    takes an explicit snapshot and execute() only ever consults the plan it
    was given, so there's no way for a config change happening between the
    two calls to affect an execute() already in flight (the class of bug a
    shared, internally-read config_store used to allow - see
    app.rag_services.rag_runtime_config)."""
    local = _FakeReranker("local")
    voyage = _FakeReranker("voyage")
    reranker = DynamicReranker(local=local, voyage=voyage, metrics=RagMetricsService())

    plan = reranker.plan("q", _config(backend="local"), enabled=True)
    # A "later" config (backend flipped to voyage, emergency turned on) is
    # never passed to execute() - only `plan` is - so it can't matter here.
    later_config = _config(backend="voyage", emergency_disabled=True)
    assert later_config.reranker_backend == "voyage"  # sanity: genuinely different

    outcome = reranker.execute(plan, query="q", candidates=_candidates(), top_k=1)

    assert outcome.backend == "local"
    assert outcome.applied is True
    assert local.calls == 1
    assert voyage.calls == 0
