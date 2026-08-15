from __future__ import annotations

from typing import cast

import pytest
from qdrant_client.models import SparseVector

from app.core.llm.sparse_embedding_client import SparseEmbeddingClient
from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.retrieval_strategy import (
    DenseRetrievalStrategy,
    HybridRetrievalStrategy,
    SparseRetrievalStrategy,
)
from app.repositories.vector_repository import VectorRepository


class _FakeVectorRepository:
    def __init__(
        self,
        dense_results: list[RetrievedChunk] | None = None,
        sparse_results: list[RetrievedChunk] | None = None,
        hybrid_results: list[RetrievedChunk] | None = None,
    ) -> None:
        self._dense_results = dense_results or []
        self._sparse_results = sparse_results or []
        self._hybrid_results = hybrid_results or []
        self.search_dense_calls: list[dict[str, object]] = []
        self.search_sparse_calls: list[dict[str, object]] = []
        self.search_hybrid_calls: list[dict[str, object]] = []

    def upsert_chunks(self, chunks, dense_embeddings, sparse_embeddings) -> None:
        raise NotImplementedError

    def search_dense(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        self.search_dense_calls.append({"query_embedding": query_embedding, "top_k": top_k})
        return self._dense_results[:top_k]

    def search_sparse(self, query_sparse: SparseVector, top_k: int = 5) -> list[RetrievedChunk]:
        self.search_sparse_calls.append({"query_sparse": query_sparse, "top_k": top_k})
        return self._sparse_results[:top_k]

    def search_hybrid(
        self,
        query_embedding: list[float],
        query_sparse: SparseVector,
        top_k: int = 5,
        candidate_top_k: int = 20,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        self.search_hybrid_calls.append(
            {
                "query_embedding": query_embedding,
                "query_sparse": query_sparse,
                "top_k": top_k,
                "candidate_top_k": candidate_top_k,
                "rrf_k": rrf_k,
            }
        )
        return self._hybrid_results[:top_k]

    def scroll_all_chunks(self, limit: int = 10_000):
        raise NotImplementedError

    def delete_by_source(self, source: str) -> None:
        raise NotImplementedError


class _FakeSparseEmbeddingClient:
    def __init__(self) -> None:
        self.query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        raise NotImplementedError

    def embed_query(self, text: str) -> SparseVector:
        self.query_calls.append(text)
        return SparseVector(indices=[1, 2], values=[1.0, 1.0])


def test_dense_strategy_delegates_to_vector_repository_search_dense() -> None:
    dense_results = [RetrievedChunk(text="dense", source="a.pdf", score=0.9)]
    fake = _FakeVectorRepository(dense_results=dense_results)
    strategy = DenseRetrievalStrategy(vector_repository=cast(VectorRepository, fake))

    results = strategy.retrieve(query_text="ignored", query_embedding=[0.1], top_k=1)

    assert results == dense_results
    assert fake.search_dense_calls == [{"query_embedding": [0.1], "top_k": 1}]


def test_dense_strategy_name_and_cache_namespace() -> None:
    strategy = DenseRetrievalStrategy(vector_repository=cast(VectorRepository, object()))

    assert strategy.name == "dense"
    assert strategy.cache_namespace == "dense:v1"


def test_sparse_strategy_encodes_query_and_delegates_to_search_sparse() -> None:
    sparse_results = [RetrievedChunk(text="sparse", source="b.pdf", score=0.5)]
    fake = _FakeVectorRepository(sparse_results=sparse_results)
    sparse_embedder = _FakeSparseEmbeddingClient()
    strategy = SparseRetrievalStrategy(
        vector_repository=cast(VectorRepository, fake),
        sparse_embedding_client=cast(SparseEmbeddingClient, sparse_embedder),
    )

    results = strategy.retrieve(query_text="refund policy", query_embedding=[0.1], top_k=3)

    assert results == sparse_results
    assert sparse_embedder.query_calls == ["refund policy"]
    assert fake.search_sparse_calls[0]["top_k"] == 3


def test_sparse_strategy_name_and_cache_namespace() -> None:
    strategy = SparseRetrievalStrategy(
        vector_repository=cast(VectorRepository, object()),
        sparse_embedding_client=cast(SparseEmbeddingClient, object()),
    )

    assert strategy.name == "sparse"
    assert strategy.cache_namespace == "sparse:v1"


def _hybrid_strategy(
    hybrid_results: list[RetrievedChunk] | None = None,
    **kwargs: object,
) -> tuple[HybridRetrievalStrategy, _FakeVectorRepository, _FakeSparseEmbeddingClient]:
    fake_repo = _FakeVectorRepository(hybrid_results=hybrid_results)
    fake_sparse = _FakeSparseEmbeddingClient()
    strategy = HybridRetrievalStrategy(
        vector_repository=cast(VectorRepository, fake_repo),
        sparse_embedding_client=cast(SparseEmbeddingClient, fake_sparse),
        **kwargs,
    )
    return strategy, fake_repo, fake_sparse


def test_hybrid_strategy_name_and_cache_namespace() -> None:
    strategy, _, _ = _hybrid_strategy()

    assert strategy.name == "hybrid"
    # v2: bumped when the underlying implementation moved from in-process
    # TF-IDF fusion to Qdrant-native fusion, so a stale cached answer from
    # the old pipeline can't be served under the new one's cache key.
    assert strategy.cache_namespace == "hybrid:v2"


def test_hybrid_search_encodes_query_and_delegates_to_search_hybrid() -> None:
    strategy, fake_repo, fake_sparse = _hybrid_strategy()

    strategy.retrieve(query_text="refund policy", query_embedding=[0.1, 0.2], top_k=5)

    assert fake_sparse.query_calls == ["refund policy"]
    call = fake_repo.search_hybrid_calls[0]
    assert call["query_embedding"] == [0.1, 0.2]
    assert call["top_k"] == 5


def test_hybrid_strategy_widens_candidate_pool_to_configured_minimum() -> None:
    strategy, fake_repo, _ = _hybrid_strategy(candidate_top_k=20)

    strategy.retrieve(query_text="query", query_embedding=[0.1], top_k=2)

    assert fake_repo.search_hybrid_calls[0]["candidate_top_k"] == 20


def test_hybrid_strategy_passes_configured_rrf_k_through() -> None:
    strategy, fake_repo, _ = _hybrid_strategy(rrf_k=42)

    strategy.retrieve(query_text="query", query_embedding=[0.1], top_k=5)

    assert fake_repo.search_hybrid_calls[0]["rrf_k"] == 42


def test_hybrid_search_normalizes_max_possible_rrf_score_to_one() -> None:
    """Qdrant's RRF uses 0-indexed ranks (score = sum of 1/(k+rank)) -
    confirmed empirically against a live server during planning. A chunk
    ranked first in both the dense and sparse prefetch branches gets the
    maximum possible score, 2/k - normalization must map that to 1.0."""
    rrf_k = 60
    max_possible_score = 2.0 / rrf_k
    raw_chunk = RetrievedChunk(text="top", source="a.pdf", score=max_possible_score)
    strategy, _, _ = _hybrid_strategy(hybrid_results=[raw_chunk], rrf_k=rrf_k)

    results = strategy.retrieve(query_text="query", query_embedding=[0.1], top_k=1)

    assert results[0].score == pytest.approx(1.0)


def test_hybrid_search_clamps_normalized_score_to_one() -> None:
    rrf_k = 60
    # Deliberately above the theoretical max, in case of floating-point
    # slop from the server - normalization must still clamp to 1.0.
    raw_chunk = RetrievedChunk(text="top", source="a.pdf", score=(2.0 / rrf_k) * 1.5)
    strategy, _, _ = _hybrid_strategy(hybrid_results=[raw_chunk], rrf_k=rrf_k)

    results = strategy.retrieve(query_text="query", query_embedding=[0.1], top_k=1)

    assert results[0].score == pytest.approx(1.0)


def test_hybrid_search_scales_lower_scores_proportionally() -> None:
    rrf_k = 60
    max_possible_score = 2.0 / rrf_k
    half_score_chunk = RetrievedChunk(text="mid", source="a.pdf", score=max_possible_score / 2)
    strategy, _, _ = _hybrid_strategy(hybrid_results=[half_score_chunk], rrf_k=rrf_k)

    results = strategy.retrieve(query_text="query", query_embedding=[0.1], top_k=1)

    assert results[0].score == pytest.approx(0.5)
