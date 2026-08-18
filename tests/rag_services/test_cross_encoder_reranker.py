from __future__ import annotations

import math

import pytest

from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.reranker.cross_encoder_reranker import LocalCrossEncoderReranker


class _FakeModel:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.predict_calls: list[dict[str, object]] = []

    def predict(self, pairs, batch_size: int, show_progress_bar: bool) -> list[float]:
        self.predict_calls.append({"pairs": pairs, "batch_size": batch_size})
        return self._scores


def _chunks(n: int) -> list[RetrievedChunk]:
    return [RetrievedChunk(text=f"t{i}", source=f"s{i}.pdf", score=0.5) for i in range(n)]


def test_construction_does_not_load_the_model() -> None:
    load_calls: list[str] = []
    LocalCrossEncoderReranker(
        model_name="some-model", model_factory=lambda name: load_calls.append(name)
    )

    assert load_calls == []


def test_model_loads_lazily_and_only_once(monkeypatch) -> None:
    load_calls: list[str] = []
    model = _FakeModel([0.1, 0.9])

    def _factory(name: str) -> _FakeModel:
        load_calls.append(name)
        return model

    reranker = LocalCrossEncoderReranker(model_name="cross-encoder/x", model_factory=_factory)
    chunks = _chunks(2)

    reranker.rerank(query="q", candidates=chunks, top_k=2)
    reranker.rerank(query="q", candidates=chunks, top_k=2)

    assert load_calls == ["cross-encoder/x"]


def test_rerank_orders_by_score_descending() -> None:
    model = _FakeModel([0.2, 0.9, 0.5])
    reranker = LocalCrossEncoderReranker(model_name="m", model_factory=lambda name: model)
    chunks = _chunks(3)

    outcome = reranker.rerank(query="q", candidates=chunks, top_k=3)

    assert [item.chunk for item in outcome.items] == [chunks[1], chunks[2], chunks[0]]
    assert [item.rerank_score for item in outcome.items] == [0.9, 0.5, 0.2]
    assert [item.original_rank for item in outcome.items] == [2, 3, 1]
    assert outcome.applied is True


def test_rerank_ties_break_by_original_order() -> None:
    model = _FakeModel([0.5, 0.5])
    reranker = LocalCrossEncoderReranker(model_name="m", model_factory=lambda name: model)
    chunks = _chunks(2)

    outcome = reranker.rerank(query="q", candidates=chunks, top_k=2)

    assert [item.chunk for item in outcome.items] == chunks


def test_rerank_truncates_to_top_k() -> None:
    model = _FakeModel([0.1, 0.9, 0.5])
    reranker = LocalCrossEncoderReranker(model_name="m", model_factory=lambda name: model)

    outcome = reranker.rerank(query="q", candidates=_chunks(3), top_k=1)

    assert len(outcome.items) == 1
    assert outcome.items[0].rerank_score == 0.9


def test_rerank_on_empty_candidates_short_circuits_without_loading_model() -> None:
    load_calls: list[str] = []
    reranker = LocalCrossEncoderReranker(
        model_name="m", model_factory=lambda name: load_calls.append(name)
    )

    outcome = reranker.rerank(query="q", candidates=[], top_k=5)

    assert outcome.items == ()
    assert outcome.applied is True
    assert load_calls == []


def test_rerank_rejects_non_finite_scores() -> None:
    model = _FakeModel([math.inf])
    reranker = LocalCrossEncoderReranker(model_name="m", model_factory=lambda name: model)

    with pytest.raises(RuntimeError):
        reranker.rerank(query="q", candidates=_chunks(1), top_k=1)


def test_rerank_rejects_non_positive_top_k() -> None:
    reranker = LocalCrossEncoderReranker(model_name="m", model_factory=lambda name: _FakeModel([]))

    with pytest.raises(ValueError):
        reranker.rerank(query="q", candidates=_chunks(1), top_k=0)


def test_cache_namespace_includes_backend_and_model_name() -> None:
    reranker = LocalCrossEncoderReranker(model_name="my-model", model_factory=lambda name: None)

    assert reranker.cache_namespace == "reranker:local:my-model:v1"
    assert reranker.name == "local-cross-encoder"
