from __future__ import annotations

from app.rag_services.dynamic_query_transformer import DynamicQueryTransformer
from app.rag_services.dynamic_reranker import DynamicReranker
from app.rag_services.query_transformer import QueryTransformOutcome
from app.rag_services.reranker import ReRankOutcome
from app.services.rag_metrics_service import RagMetricsService


class _FakeReranker:
    """Mirrors ``tests/rag_services/test_dynamic_reranker.py``'s fake -
    unlike ``NoOpReranker``, this reports ``applied=True`` when actually
    invoked, so cohort membership is observable from the outcome."""

    @property
    def name(self) -> str:
        return "fake"

    @property
    def cache_namespace(self) -> str:
        return "reranker:fake:v1"

    def rerank(self, *, query: str, candidates: object, top_k: int) -> ReRankOutcome:
        return ReRankOutcome(items=(), backend="fake", applied=True)


class _FakeDelegate:
    def __init__(self, *, raise_error: bool = False) -> None:
        self._raise_error = raise_error
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return "hyde"

    @property
    def cache_namespace(self) -> str:
        return "hyde:v1"

    def transform(self, query: str) -> QueryTransformOutcome:
        self.calls.append(query)
        if self._raise_error:
            raise RuntimeError("boom")
        return QueryTransformOutcome(
            retrieval_texts=("a hypothetical passage",), backend="hyde", applied=True,
            usage_tokens=7,
        )


def test_zero_percent_rollout_never_calls_the_delegate() -> None:
    delegate = _FakeDelegate()
    transformer = DynamicQueryTransformer(
        delegate=delegate, metrics=RagMetricsService(), rollout_percentage=0
    )

    outcome = transformer.transform("any question")

    assert outcome.applied is False
    assert outcome.bypass_reason == "rollout"
    assert delegate.calls == []


def test_hundred_percent_rollout_always_calls_the_delegate() -> None:
    delegate = _FakeDelegate()
    transformer = DynamicQueryTransformer(
        delegate=delegate, metrics=RagMetricsService(), rollout_percentage=100
    )

    for question in ["a", "b", "c", "d", "e"]:
        transformer.transform(question)

    assert delegate.calls == ["a", "b", "c", "d", "e"]


def test_rollout_sampling_is_deterministic_for_the_same_normalized_query() -> None:
    delegate = _FakeDelegate()
    transformer = DynamicQueryTransformer(
        delegate=delegate, metrics=RagMetricsService(), rollout_percentage=50
    )

    first_run = [transformer.transform(f"question {i}").applied for i in range(20)]
    second_run = [transformer.transform(f"question {i}").applied for i in range(20)]

    assert first_run == second_run


def test_hyde_and_reranker_cohorts_are_not_forced_identical_by_a_shared_salt() -> None:
    """Both use the same underlying sampled_in() hashing, but with
    feature-specific salts - so the same rollout_percentage on the same
    question set doesn't have to select the exact same population."""
    hyde = DynamicQueryTransformer(
        delegate=_FakeDelegate(), metrics=RagMetricsService(), rollout_percentage=50
    )
    reranker = DynamicReranker(
        local=_FakeReranker(), voyage=None, metrics=RagMetricsService(), rollout_percentage=50
    )

    questions = [f"question {i}" for i in range(50)]
    hyde_sampled_in = {q for q in questions if hyde.transform(q).applied}
    reranker_sampled_in = {
        q for q in questions if reranker.rerank(query=q, candidates=[], top_k=1).applied
    }

    assert hyde_sampled_in != reranker_sampled_in


def test_emergency_disabled_bypasses_the_delegate_entirely() -> None:
    delegate = _FakeDelegate()
    transformer = DynamicQueryTransformer(
        delegate=delegate,
        metrics=RagMetricsService(),
        rollout_percentage=100,
        emergency_disabled=True,
    )

    outcome = transformer.transform("q")

    assert outcome.applied is False
    assert outcome.bypass_reason == "emergency_disabled"
    assert delegate.calls == []


def test_configure_swaps_rollout_and_emergency_as_one_atomic_state() -> None:
    delegate = _FakeDelegate()
    transformer = DynamicQueryTransformer(
        delegate=delegate, metrics=RagMetricsService(), rollout_percentage=0
    )

    transformer.configure(rollout_percentage=100, emergency_disabled=False)
    outcome = transformer.transform("q")

    assert outcome.applied is True
    assert delegate.calls == ["q"]


def test_configure_emergency_disable_overrides_rollout() -> None:
    delegate = _FakeDelegate()
    transformer = DynamicQueryTransformer(
        delegate=delegate, metrics=RagMetricsService(), rollout_percentage=100
    )

    transformer.configure(rollout_percentage=100, emergency_disabled=True)
    outcome = transformer.transform("q")

    assert outcome.applied is False
    assert outcome.bypass_reason == "emergency_disabled"


def test_delegate_failure_fails_open_and_is_recorded_as_an_attempt_not_a_bypass() -> None:
    """DynamicQueryTransformer wraps its delegate in FailOpenQueryTransformer
    internally - a raw provider/schema error must never propagate out of
    transform() and must never produce a 5xx (see RAGService.answer)."""
    delegate = _FakeDelegate(raise_error=True)
    metrics = RagMetricsService()
    transformer = DynamicQueryTransformer(
        delegate=delegate, metrics=metrics, rollout_percentage=100
    )

    outcome = transformer.transform("q")

    assert outcome.applied is False
    assert outcome.fallback is True
    stats = metrics.hyde_stats()
    assert stats.sample_count == 1
    assert stats.fallback_rate == 1.0
    assert stats.rollout_bypasses == 0
    assert stats.emergency_bypasses == 0


def test_attempt_and_bypass_metrics_are_recorded_exactly_once_each() -> None:
    metrics = RagMetricsService()
    transformer = DynamicQueryTransformer(
        delegate=_FakeDelegate(), metrics=metrics, rollout_percentage=100
    )

    transformer.transform("attempted")

    stats = metrics.hyde_stats()
    assert stats.sample_count == 1
    assert stats.usage_tokens_total == 7

    transformer.configure(rollout_percentage=0, emergency_disabled=False)
    transformer.transform("bypassed by rollout")

    stats = metrics.hyde_stats()
    assert stats.sample_count == 1  # unchanged - a bypass is not an attempt
    assert stats.rollout_bypasses == 1

    transformer.configure(rollout_percentage=100, emergency_disabled=True)
    transformer.transform("bypassed by emergency")

    stats = metrics.hyde_stats()
    assert stats.sample_count == 1
    assert stats.emergency_bypasses == 1


def test_cache_namespace_changes_when_rollout_or_emergency_state_changes() -> None:
    transformer = DynamicQueryTransformer(
        delegate=_FakeDelegate(), metrics=RagMetricsService(), rollout_percentage=0
    )

    baseline = transformer.cache_namespace
    transformer.configure(rollout_percentage=50, emergency_disabled=False)
    after_rollout_change = transformer.cache_namespace
    transformer.configure(rollout_percentage=50, emergency_disabled=True)
    after_emergency_change = transformer.cache_namespace

    assert baseline != after_rollout_change
    assert after_rollout_change != after_emergency_change
