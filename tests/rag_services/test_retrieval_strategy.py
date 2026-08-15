from __future__ import annotations

from typing import cast

import pytest

from app.core.ingestion.document_processor import DocumentChunk
from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.retrieval_strategy import DenseRetrievalStrategy, HybridRetrievalStrategy
from app.rag_services.sparse_vector_service import SparseVectorIndex
from app.repositories.vector_repository import VectorRepository


class _FakeVectorRepository:
    def __init__(
        self, dense_results: list[RetrievedChunk], documents: list[dict[str, str]]
    ) -> None:
        self._dense_results = dense_results
        self._documents = documents
        self.search_calls: list[dict[str, object]] = []

    def upsert_chunks(
        self, chunks: list[DocumentChunk], embeddings: list[list[float]]
    ) -> None:
        raise NotImplementedError

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        self.search_calls.append({"query_embedding": query_embedding, "top_k": top_k})
        return self._dense_results[:top_k]

    def scroll_all_chunks(self, limit: int = 10_000) -> list[dict[str, str]]:
        return self._documents


def _hybrid_strategy(
    dense_results: list[RetrievedChunk], documents: list[dict[str, str]]
) -> tuple[HybridRetrievalStrategy, _FakeVectorRepository]:
    fake = _FakeVectorRepository(dense_results, documents)
    return HybridRetrievalStrategy(vector_repository=cast(VectorRepository, fake)), fake


def test_dense_strategy_delegates_to_vector_repository_search() -> None:
    dense_results = [RetrievedChunk(text="dense", source="a.pdf", score=0.9)]
    fake = _FakeVectorRepository(dense_results, documents=[])
    strategy = DenseRetrievalStrategy(vector_repository=cast(VectorRepository, fake))

    results = strategy.retrieve(query_text="ignored", query_embedding=[0.1], top_k=1)

    assert results == dense_results
    assert fake.search_calls == [{"query_embedding": [0.1], "top_k": 1}]


def test_dense_strategy_name_and_cache_namespace() -> None:
    strategy = DenseRetrievalStrategy(vector_repository=cast(VectorRepository, object()))

    assert strategy.name == "dense"
    assert strategy.cache_namespace == "dense:v1"


def test_hybrid_strategy_name_and_cache_namespace() -> None:
    strategy, _ = _hybrid_strategy(dense_results=[], documents=[])

    assert strategy.name == "hybrid"
    assert strategy.cache_namespace == "hybrid:v1"


def test_hybrid_search_builds_sparse_index_from_vector_repository_scroll() -> None:
    documents = [
        {"id": "1", "text": "refund policy for wire transfers", "source": "a.pdf"},
        {"id": "2", "text": "branch holiday hours", "source": "b.pdf"},
    ]
    strategy, _ = _hybrid_strategy(dense_results=[], documents=documents)

    results = strategy.hybrid_search(query_embedding=[0.1], query_text="refund policy", top_k=5)

    assert len(results) == 1
    assert results[0].source == "a.pdf"


def test_hybrid_search_uses_sparse_top_k_not_rrf_k_for_sparse_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: hybrid_search must pass ``sparse_top_k`` into the
    sparse half of the search, not ``rrf_k`` - narrowing the sparse
    candidate pool to an unrelated parameter would silently drop hits."""
    captured_top_k: list[int] = []
    original_search = SparseVectorIndex.search

    def spy_search(self: SparseVectorIndex, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        captured_top_k.append(top_k)
        return original_search(self, query, top_k=top_k)

    monkeypatch.setattr(SparseVectorIndex, "search", spy_search)
    documents = [{"id": "1", "text": "refund policy content", "source": "a.pdf"}]
    strategy, _ = _hybrid_strategy(dense_results=[], documents=documents)

    strategy.hybrid_search(
        query_embedding=[0.1], query_text="refund policy", rrf_k=3, sparse_top_k=17
    )

    assert captured_top_k == [17]


def test_hybrid_search_passes_sparse_top_k_to_dense_search() -> None:
    strategy, fake = _hybrid_strategy(dense_results=[], documents=[])

    strategy.hybrid_search(query_embedding=[0.1], query_text="query", sparse_top_k=17)

    assert fake.search_calls[0]["top_k"] == 17


def test_hybrid_search_truncates_final_results_to_top_k() -> None:
    dense_results = [RetrievedChunk(text=f"dense {i}", source="a.pdf", score=1.0) for i in range(5)]
    documents = [{"id": str(i), "text": f"sparse {i} content", "source": "b.pdf"} for i in range(5)]
    strategy, _ = _hybrid_strategy(dense_results, documents)

    results = strategy.hybrid_search(query_embedding=[0.1], query_text="content", top_k=2)

    assert len(results) <= 2


def test_hybrid_search_normalizes_scores_into_zero_one_range() -> None:
    """A chunk ranked #1 in both the dense and sparse lists gets the
    maximum possible RRF score - normalization must map that to 1.0 so
    hybrid scores stay comparable to dense cosine scores (see
    ``confidence_scorer._RELEVANCE_FLOOR``, calibrated for a [0, 1] scale)."""
    shared_text = "refund policy for wire transfers"
    dense_results = [RetrievedChunk(text=shared_text, source="a.pdf", score=0.42)]
    documents = [{"id": "1", "text": shared_text, "source": "a.pdf"}]
    strategy, _ = _hybrid_strategy(dense_results, documents)

    results = strategy.hybrid_search(
        query_embedding=[0.1], query_text="refund policy wire transfers", top_k=1, rrf_k=60
    )

    assert results[0].score == pytest.approx(1.0)


def test_hybrid_strategy_retrieve_widens_candidate_pool_to_configured_minimum() -> None:
    fake = _FakeVectorRepository(dense_results=[], documents=[])
    strategy = HybridRetrievalStrategy(
        vector_repository=cast(VectorRepository, fake), candidate_top_k=20
    )

    strategy.retrieve(query_text="query", query_embedding=[0.1], top_k=2)

    assert fake.search_calls[0]["top_k"] == 20
