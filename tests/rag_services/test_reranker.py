from __future__ import annotations

import pytest

from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.reranker import FailOpenReranker, NoOpReranker, ReRankedChunk, ReRankOutcome


class _FakeReranker:
    def __init__(self, *, name: str = "fake", raises: bool = False) -> None:
        self._name = name
        self._raises = raises
        self.calls: list[dict[str, object]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def cache_namespace(self) -> str:
        return f"reranker:{self._name}:v1"

    def rerank(self, *, query: str, candidates, top_k: int) -> ReRankOutcome:
        self.calls.append({"query": query, "candidates": candidates, "top_k": top_k})
        if self._raises:
            raise RuntimeError("boom")
        items = tuple(
            ReRankedChunk(chunk=chunk, original_rank=rank, rerank_score=1.0)
            for rank, chunk in enumerate(candidates[:top_k], start=1)
        )
        return ReRankOutcome(items=items, backend=self._name, applied=True)


def _chunks(n: int) -> list[RetrievedChunk]:
    return [RetrievedChunk(text=f"t{i}", source=f"s{i}.pdf", score=1.0 - i * 0.1) for i in range(n)]


def test_noop_reranker_preserves_retrieval_order_and_null_scores() -> None:
    chunks = _chunks(3)

    outcome = NoOpReranker().rerank(query="q", candidates=chunks, top_k=2)

    assert [item.chunk for item in outcome.items] == chunks[:2]
    assert [item.original_rank for item in outcome.items] == [1, 2]
    assert all(item.rerank_score is None for item in outcome.items)
    assert outcome.applied is False
    assert outcome.backend == "none"


def test_noop_reranker_rejects_non_positive_top_k() -> None:
    with pytest.raises(ValueError):
        NoOpReranker().rerank(query="q", candidates=_chunks(1), top_k=0)


def test_fail_open_reranker_returns_delegate_result_on_success() -> None:
    delegate = _FakeReranker()
    reranker = FailOpenReranker(delegate=delegate)

    outcome = reranker.rerank(query="q", candidates=_chunks(3), top_k=2)

    assert len(outcome.items) == 2
    assert outcome.fallback is False
    assert reranker.name == delegate.name
    assert reranker.cache_namespace == f"fail-open:v1:{delegate.cache_namespace}"


def test_fail_open_reranker_falls_back_to_retrieval_order_without_a_fallback_reranker() -> None:
    delegate = _FakeReranker(raises=True)
    chunks = _chunks(3)
    reranker = FailOpenReranker(delegate=delegate)

    outcome = reranker.rerank(query="q", candidates=chunks, top_k=2)

    assert [item.chunk for item in outcome.items] == chunks[:2]
    assert all(item.rerank_score is None for item in outcome.items)
    assert outcome.applied is False
    assert outcome.fallback is True


def test_fail_open_reranker_uses_fallback_reranker_on_delegate_failure() -> None:
    delegate = _FakeReranker(name="primary", raises=True)
    fallback = _FakeReranker(name="secondary")
    chunks = _chunks(3)
    reranker = FailOpenReranker(delegate=delegate, fallback=fallback)

    outcome = reranker.rerank(query="q", candidates=chunks, top_k=2)

    assert len(fallback.calls) == 1
    assert [item.chunk for item in outcome.items] == chunks[:2]
    assert outcome.applied is False
    assert outcome.fallback is True
