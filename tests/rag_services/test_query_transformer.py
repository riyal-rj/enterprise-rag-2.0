from __future__ import annotations

from app.rag_services.hyde.query_transformer import (
    FailOpenQueryTransformer,
    NoOpQueryTransformer,
    PlannedNoOpQueryTransformer,
    QueryTransformOutcome,
    StaticPlannedQueryTransformer,
)
from app.rag_services.rag_runtime_config import RagRuntimeConfig


class _FakeTransformer:
    def __init__(self, *, raise_error: bool = False) -> None:
        self._raise_error = raise_error
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def cache_namespace(self) -> str:
        return "fake:v1"

    def transform(self, query: str) -> QueryTransformOutcome:
        self.calls.append(query)
        if self._raise_error:
            raise RuntimeError("boom")
        return QueryTransformOutcome(
            retrieval_texts=("hypothesis one",), backend="fake", applied=True
        )


def test_noop_transformer_never_applies_and_returns_the_original_query() -> None:
    transformer = NoOpQueryTransformer(reason="disabled")

    outcome = transformer.transform("what is the policy?")

    assert outcome.applied is False
    assert outcome.retrieval_texts == ("what is the policy?",)
    assert outcome.bypass_reason == "disabled"


def test_noop_transformer_cache_namespace_reflects_the_reason() -> None:
    disabled = NoOpQueryTransformer(reason="disabled")
    rollout = NoOpQueryTransformer(reason="rollout")

    assert disabled.cache_namespace != rollout.cache_namespace


def test_fail_open_transformer_passes_through_a_successful_delegate() -> None:
    delegate = _FakeTransformer()
    transformer = FailOpenQueryTransformer(delegate)

    outcome = transformer.transform("q")

    assert outcome.applied is True
    assert outcome.retrieval_texts == ("hypothesis one",)
    assert delegate.calls == ["q"]


def test_fail_open_transformer_converts_delegate_errors_to_original_query_fallback() -> None:
    delegate = _FakeTransformer(raise_error=True)
    transformer = FailOpenQueryTransformer(delegate)

    outcome = transformer.transform("what is the policy?")

    assert outcome.applied is False
    assert outcome.fallback is True
    assert outcome.retrieval_texts == ("what is the policy?",)
    assert outcome.backend == "fake"


def test_fail_open_transformer_exposes_the_delegate_name_and_namespace() -> None:
    delegate = _FakeTransformer()
    transformer = FailOpenQueryTransformer(delegate)

    assert transformer.name == "fake"
    assert transformer.cache_namespace == "fake:v1"


def _config() -> RagRuntimeConfig:
    return RagRuntimeConfig(
        reranking_enabled=False,
        reranker_backend="local",
        reranker_rollout_percentage=100,
        emergency_disabled=False,
        semantic_cache_enabled=False,
        semantic_cache_threshold=0.95,
        corpus_version=1,
        hyde_enabled=True,
        hyde_rollout_percentage=100,
        crag_enabled=False,
        crag_rollout_percentage=0,
        crag_web_enabled=False,
    )


def test_planned_noop_query_transformer_is_always_disabled() -> None:
    plan = PlannedNoOpQueryTransformer().plan("q", _config(), enabled=True)

    assert plan.cohort == "disabled"


def test_planned_noop_query_transformer_execute_is_a_noop() -> None:
    transformer = PlannedNoOpQueryTransformer()
    plan = transformer.plan("q", _config(), enabled=True)

    outcome = transformer.execute("q", plan)

    assert outcome.applied is False


def test_static_planned_query_transformer_is_disabled_when_not_enabled() -> None:
    transformer = StaticPlannedQueryTransformer(_FakeTransformer())

    plan = transformer.plan("q", _config(), enabled=False)

    assert plan.cohort == "disabled"


def test_static_planned_query_transformer_is_always_treatment_when_enabled_regardless_of_rollout() -> (
    None
):
    """Mirrors get_eval_hyde_transformer()'s docstring guarantee: eval must
    exercise HyDE for real whenever enable_hyde is on, never silently
    bypassed by an admin's live rollout%/emergency-disable setting - even a
    0%-rollout, emergency-disabled config must not change this."""
    delegate = _FakeTransformer()
    transformer = StaticPlannedQueryTransformer(delegate)
    hostile_config = RagRuntimeConfig(
        reranking_enabled=False,
        reranker_backend="local",
        reranker_rollout_percentage=100,
        emergency_disabled=True,
        semantic_cache_enabled=False,
        semantic_cache_threshold=0.95,
        corpus_version=1,
        hyde_enabled=True,
        hyde_rollout_percentage=0,
        crag_enabled=False,
        crag_rollout_percentage=0,
        crag_web_enabled=False,
    )

    plan = transformer.plan("q", hostile_config, enabled=True)
    outcome = transformer.execute("q", plan)

    assert plan.cohort == "treatment"
    assert outcome.applied is True
    assert delegate.calls == ["q"]
