from __future__ import annotations

from app.rag_services.dynamic_query_transformer import DynamicQueryTransformer
from app.rag_services.dynamic_reranker import DynamicReranker
from app.rag_services.query_transformer import QueryTransformOutcome
from app.rag_services.rag_runtime_config import RagRuntimeConfig
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
            retrieval_texts=("a hypothetical passage",),
            backend="hyde",
            applied=True,
            usage_tokens=7,
        )


def _config(*, rollout_percentage: int = 100, emergency_disabled: bool = False) -> RagRuntimeConfig:
    return RagRuntimeConfig(
        reranking_enabled=False,
        reranker_backend="local",
        reranker_rollout_percentage=100,
        emergency_disabled=emergency_disabled,
        semantic_cache_enabled=False,
        semantic_cache_threshold=0.95,
        corpus_version=1,
        hyde_enabled=True,
        hyde_rollout_percentage=rollout_percentage,
        crag_enabled=False,
        crag_rollout_percentage=0,
        crag_web_enabled=False,
    )


def test_plan_is_disabled_cohort_when_not_enabled() -> None:
    transformer = DynamicQueryTransformer(delegate=_FakeDelegate())

    plan = transformer.plan("any question", _config(), enabled=False)

    assert plan.cohort == "disabled"
    assert plan.bypass_reason == "disabled"


def test_zero_percent_rollout_is_control_cohort_and_never_calls_the_delegate() -> None:
    delegate = _FakeDelegate()
    transformer = DynamicQueryTransformer(delegate=delegate)

    plan = transformer.plan("any question", _config(rollout_percentage=0), enabled=True)
    outcome = transformer.execute("any question", plan)

    assert plan.cohort == "control"
    assert plan.bypass_reason == "rollout"
    assert outcome.applied is False
    assert outcome.bypass_reason == "rollout"
    assert delegate.calls == []


def test_hundred_percent_rollout_is_always_treatment_and_calls_the_delegate() -> None:
    delegate = _FakeDelegate()
    transformer = DynamicQueryTransformer(delegate=delegate)
    config = _config(rollout_percentage=100)

    for question in ["a", "b", "c", "d", "e"]:
        plan = transformer.plan(question, config, enabled=True)
        transformer.execute(question, plan)

    assert delegate.calls == ["a", "b", "c", "d", "e"]


def test_rollout_sampling_is_deterministic_for_the_same_normalized_query() -> None:
    transformer = DynamicQueryTransformer(delegate=_FakeDelegate())
    config = _config(rollout_percentage=50)

    first_run = [transformer.plan(f"question {i}", config, enabled=True).cohort for i in range(20)]
    second_run = [transformer.plan(f"question {i}", config, enabled=True).cohort for i in range(20)]

    assert first_run == second_run


def test_hyde_and_reranker_cohorts_are_not_forced_identical_by_a_shared_salt() -> None:
    """Both use the same underlying sampled_in() hashing, but with
    feature-specific salts - so the same rollout_percentage on the same
    question set doesn't have to select the exact same population."""
    hyde = DynamicQueryTransformer(delegate=_FakeDelegate())
    reranker = DynamicReranker(local=_FakeReranker(), voyage=None, metrics=RagMetricsService())

    hyde_config = RagRuntimeConfig(
        reranking_enabled=True,
        reranker_backend="local",
        reranker_rollout_percentage=50,
        emergency_disabled=False,
        semantic_cache_enabled=False,
        semantic_cache_threshold=0.95,
        corpus_version=1,
        hyde_enabled=True,
        hyde_rollout_percentage=50,
        crag_enabled=False,
        crag_rollout_percentage=0,
        crag_web_enabled=False,
    )

    questions = [f"question {i}" for i in range(50)]
    hyde_sampled_in = {
        q for q in questions if hyde.plan(q, hyde_config, enabled=True).cohort == "treatment"
    }
    reranker_sampled_in = {
        q for q in questions if reranker.plan(q, hyde_config, enabled=True).cohort == "treatment"
    }

    assert hyde_sampled_in != reranker_sampled_in


def test_emergency_disabled_bypasses_the_delegate_entirely() -> None:
    delegate = _FakeDelegate()
    transformer = DynamicQueryTransformer(delegate=delegate)

    plan = transformer.plan(
        "q", _config(rollout_percentage=100, emergency_disabled=True), enabled=True
    )
    outcome = transformer.execute("q", plan)

    assert plan.cohort == "disabled"
    assert plan.bypass_reason == "emergency_disabled"
    assert outcome.applied is False
    assert outcome.bypass_reason == "emergency_disabled"
    assert delegate.calls == []


def test_delegate_failure_fails_open() -> None:
    """DynamicQueryTransformer wraps its delegate in FailOpenQueryTransformer
    internally - a raw provider/schema error must never propagate out of
    execute() and must never produce a 5xx (see RAGService.answer)."""
    delegate = _FakeDelegate(raise_error=True)
    transformer = DynamicQueryTransformer(delegate=delegate)

    plan = transformer.plan("q", _config(rollout_percentage=100), enabled=True)
    outcome = transformer.execute("q", plan)

    assert plan.cohort == "treatment"
    assert outcome.applied is False
    assert outcome.fallback is True


def test_cache_namespace_isolates_control_and_treatment_cohorts() -> None:
    """Regression: the cache namespace used to only encode the *configured*
    rollout percentage (e.g. "rollout=50:emergency=0"), not which cohort a
    specific query actually landed in - so a control-cohort query and a
    treatment-cohort query under the same 50% rollout would share one cache
    namespace, letting a semantic-cache hit serve one query an answer
    generated under the other's (different) retrieval vector. plan()'s
    cache_namespace must instead encode the *resolved* cohort."""
    transformer = DynamicQueryTransformer(delegate=_FakeDelegate())
    config = _config(rollout_percentage=50)

    control_query = next(
        q
        for q in (f"q{i}" for i in range(50))
        if transformer.plan(q, config, enabled=True).cohort == "control"
    )
    treatment_query = next(
        q
        for q in (f"q{i}" for i in range(50))
        if transformer.plan(q, config, enabled=True).cohort == "treatment"
    )

    control_plan = transformer.plan(control_query, config, enabled=True)
    treatment_plan = transformer.plan(treatment_query, config, enabled=True)

    assert control_plan.cache_namespace != treatment_plan.cache_namespace


def test_cache_namespace_differs_for_disabled_vs_emergency_vs_rollout_bypass() -> None:
    transformer = DynamicQueryTransformer(delegate=_FakeDelegate())

    disabled = transformer.plan("q", _config(), enabled=False)
    emergency = transformer.plan("q", _config(emergency_disabled=True), enabled=True)
    rollout = transformer.plan("q", _config(rollout_percentage=0), enabled=True)

    namespaces = {disabled.cache_namespace, emergency.cache_namespace, rollout.cache_namespace}
    assert len(namespaces) == 3


def test_execute_is_driven_entirely_by_the_plan_not_by_a_later_config_change() -> None:
    """DynamicQueryTransformer holds no config reference of its own - plan()
    takes an explicit snapshot and execute() only ever consults the plan it
    was given, so a config change happening between the two calls can't
    affect an execute() already in flight."""
    delegate = _FakeDelegate()
    transformer = DynamicQueryTransformer(delegate=delegate)

    plan = transformer.plan("q", _config(rollout_percentage=100), enabled=True)
    later_config = _config(rollout_percentage=0, emergency_disabled=True)
    assert later_config.hyde_rollout_percentage == 0  # sanity: genuinely different

    outcome = transformer.execute("q", plan)

    assert outcome.applied is True
    assert delegate.calls == ["q"]
