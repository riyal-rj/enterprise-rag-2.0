from __future__ import annotations

from app.rag_services.query_transformer import (
    FailOpenQueryTransformer,
    NoOpQueryTransformer,
    QueryTransformOutcome,
)


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
