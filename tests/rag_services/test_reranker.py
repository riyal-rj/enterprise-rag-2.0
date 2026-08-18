from __future__ import annotations

import pytest

from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.rag_runtime_config import RagRuntimeConfig
from app.rag_services.reranker import (
    FailOpenReranker,
    NoOpReranker,
    PlannedNoOpReranker,
    ReRankedChunk,
    ReRankOutcome,
    StaticPlannedReranker,
)


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


def _config() -> RagRuntimeConfig:
    return RagRuntimeConfig(
        reranking_enabled=True,
        reranker_backend="local",
        reranker_rollout_percentage=100,
        emergency_disabled=False,
        semantic_cache_enabled=False,
        semantic_cache_threshold=0.95,
        corpus_version=1,
        hyde_enabled=False,
        hyde_rollout_percentage=0,
        crag_enabled=False,
        crag_rollout_percentage=0,
        crag_web_enabled=False,
    )


def test_planned_noop_reranker_is_always_disabled() -> None:
    plan = PlannedNoOpReranker().plan("q", _config(), enabled=True)

    assert plan.cohort == "disabled"
    assert plan.reported_backend == "none"


def test_planned_noop_reranker_execute_is_a_noop() -> None:
    reranker = PlannedNoOpReranker()
    plan = reranker.plan("q", _config(), enabled=True)

    outcome = reranker.execute(plan, query="q", candidates=_chunks(2), top_k=2)

    assert outcome.applied is False
    assert outcome.backend == "none"


def test_static_planned_reranker_is_disabled_when_not_enabled() -> None:
    reranker = StaticPlannedReranker(_FakeReranker())

    plan = reranker.plan("q", _config(), enabled=False)

    assert plan.cohort == "disabled"


def test_static_planned_reranker_is_always_treatment_when_enabled_regardless_of_rollout() -> None:
    """Mirrors get_reranker()'s docstring guarantee: eval must exercise
    reranking for real whenever enable_rerank is on, never silently
    bypassed by an admin's live rollout%/emergency-disable setting - even
    a 0%-rollout, emergency-disabled config must not change this."""
    delegate = _FakeReranker()
    reranker = StaticPlannedReranker(delegate)
    hostile_config = RagRuntimeConfig(
        reranking_enabled=False,
        reranker_backend="local",
        reranker_rollout_percentage=0,
        emergency_disabled=True,
        semantic_cache_enabled=False,
        semantic_cache_threshold=0.95,
        corpus_version=1,
        hyde_enabled=False,
        hyde_rollout_percentage=0,
        crag_enabled=False,
        crag_rollout_percentage=0,
        crag_web_enabled=False,
    )

    plan = reranker.plan("q", hostile_config, enabled=True)
    outcome = reranker.execute(plan, query="q", candidates=_chunks(2), top_k=2)

    assert plan.cohort == "treatment"
    assert outcome.applied is True
    assert len(delegate.calls) == 1
