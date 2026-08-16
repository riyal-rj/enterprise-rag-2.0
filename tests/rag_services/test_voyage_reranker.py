from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
import voyageai

from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.voyage_reranker import VoyageReranker


class _FakeClient:
    """Mirrors the real ``voyageai.Client.rerank`` signature and return
    shape exactly - see ``test_rerank_call_is_compatible_with_the_installed_sdk``
    for the autospecced version that fails the build the moment this drifts
    from the real SDK again (as happened when ``truncate``/``usage_tokens``
    were used instead of ``truncation``/``total_tokens``)."""

    def __init__(self, results: list[SimpleNamespace], total_tokens: int | None = None) -> None:
        self._results = results
        self._total_tokens = total_tokens
        self.rerank_calls: list[dict[str, object]] = []

    def rerank(self, query, documents, model, top_k=None, truncation=True):
        self.rerank_calls.append(
            {"query": query, "documents": documents, "model": model, "top_k": top_k}
        )
        response = SimpleNamespace(results=self._results)
        if self._total_tokens is not None:
            response.total_tokens = self._total_tokens
        return response


def _chunks(n: int) -> list[RetrievedChunk]:
    return [RetrievedChunk(text=f"t{i}", source=f"s{i}.pdf", score=0.5) for i in range(n)]


def _result(index: int, score: float) -> SimpleNamespace:
    return SimpleNamespace(index=index, relevance_score=score)


def test_rerank_call_is_compatible_with_the_installed_sdks_signature() -> None:
    """Contract test: binds our call site against the actual installed
    ``voyageai.Client.rerank`` signature via autospec, so a real SDK
    parameter rename (e.g. truncate -> truncation) fails this test
    immediately instead of only being caught by a hand-rolled fake that
    silently drifted out of sync with the real contract."""
    autospec_client = create_autospec(voyageai.Client, instance=True)
    autospec_client.rerank.return_value = SimpleNamespace(
        results=[_result(0, 0.9)], total_tokens=10
    )
    reranker = VoyageReranker(api_key="key", model_name="rerank-2.5", client=autospec_client)

    reranker.rerank(query="q", candidates=_chunks(1), top_k=1)

    autospec_client.rerank.assert_called_once()
    call_kwargs = autospec_client.rerank.call_args.kwargs
    assert call_kwargs["truncation"] is True
    assert "truncate" not in call_kwargs


def test_rerank_with_more_than_one_result_does_not_raise() -> None:
    """Regression test: the result-count check used to live inside the
    per-result loop and fired after the first item, so any multi-result
    response raised before the reranker could return anything."""
    chunks = _chunks(3)
    client = _FakeClient([_result(1, 0.9), _result(0, 0.5), _result(2, 0.1)])
    reranker = VoyageReranker(api_key="key", model_name="rerank-2.5", client=client)

    outcome = reranker.rerank(query="q", candidates=chunks, top_k=3)

    assert len(outcome.items) == 3
    assert outcome.applied is True


def test_rerank_orders_by_relevance_score_descending() -> None:
    chunks = _chunks(3)
    client = _FakeClient([_result(0, 0.2), _result(1, 0.9), _result(2, 0.5)])
    reranker = VoyageReranker(api_key="key", model_name="m", client=client)

    outcome = reranker.rerank(query="q", candidates=chunks, top_k=3)

    assert [item.chunk for item in outcome.items] == [chunks[1], chunks[2], chunks[0]]
    assert [item.rerank_score for item in outcome.items] == [0.9, 0.5, 0.2]


def test_rerank_passes_limited_top_k_and_documents_to_client() -> None:
    chunks = _chunks(3)
    client = _FakeClient([_result(0, 0.9)])
    reranker = VoyageReranker(api_key="key", model_name="rerank-2.5", client=client)

    reranker.rerank(query="what is the policy?", candidates=chunks, top_k=1)

    call = client.rerank_calls[0]
    assert call["query"] == "what is the policy?"
    assert call["documents"] == ["t0", "t1", "t2"]
    assert call["top_k"] == 1
    assert call["model"] == "rerank-2.5"


def test_rerank_surfaces_total_tokens_when_present() -> None:
    client = _FakeClient([_result(0, 0.9)], total_tokens=42)
    reranker = VoyageReranker(api_key="key", model_name="m", client=client)

    outcome = reranker.rerank(query="q", candidates=_chunks(1), top_k=1)

    assert outcome.usage_tokens == 42


def test_rerank_defaults_usage_tokens_to_none_when_total_tokens_absent() -> None:
    client = _FakeClient([_result(0, 0.9)])
    reranker = VoyageReranker(api_key="key", model_name="m", client=client)

    outcome = reranker.rerank(query="q", candidates=_chunks(1), top_k=1)

    assert outcome.usage_tokens is None


def test_rerank_on_empty_candidates_short_circuits_without_calling_client() -> None:
    client = _FakeClient([])
    reranker = VoyageReranker(api_key="key", model_name="m", client=client)

    outcome = reranker.rerank(query="q", candidates=[], top_k=5)

    assert outcome.items == ()
    assert client.rerank_calls == []


def test_rerank_rejects_out_of_range_index() -> None:
    client = _FakeClient([_result(5, 0.9)])
    reranker = VoyageReranker(api_key="key", model_name="m", client=client)

    with pytest.raises(RuntimeError, match="invalid document index"):
        reranker.rerank(query="q", candidates=_chunks(2), top_k=1)


def test_rerank_rejects_duplicate_index() -> None:
    client = _FakeClient([_result(0, 0.9), _result(0, 0.1)])
    reranker = VoyageReranker(api_key="key", model_name="m", client=client)

    with pytest.raises(RuntimeError, match="duplicate document index"):
        reranker.rerank(query="q", candidates=_chunks(2), top_k=2)


def test_rerank_rejects_non_finite_score() -> None:
    client = _FakeClient([_result(0, math.inf)])
    reranker = VoyageReranker(api_key="key", model_name="m", client=client)

    with pytest.raises(RuntimeError, match="infinite score"):
        reranker.rerank(query="q", candidates=_chunks(1), top_k=1)


def test_rerank_rejects_result_count_mismatch() -> None:
    client = _FakeClient([_result(0, 0.9)])
    reranker = VoyageReranker(api_key="key", model_name="m", client=client)

    with pytest.raises(RuntimeError, match="expected 2"):
        reranker.rerank(query="q", candidates=_chunks(3), top_k=2)


def test_constructor_rejects_empty_api_key() -> None:
    with pytest.raises(ValueError):
        VoyageReranker(api_key="", model_name="m", client=_FakeClient([]))


def test_constructor_rejects_non_positive_max_concurrency() -> None:
    with pytest.raises(ValueError):
        VoyageReranker(api_key="key", model_name="m", max_concurrency=0, client=_FakeClient([]))
