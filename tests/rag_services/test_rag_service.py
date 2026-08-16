from __future__ import annotations

import logging
from typing import cast

import pytest
from qdrant_client.models import SparseVector

from app.core.config.cache import CacheSettings
from app.core.exceptions import HybridRetrievalDisabledError
from app.core.ingestion.document_processor import DocumentChunk
from app.core.llm.chat_client import LLMClient, LLMResponse, TokenUsage
from app.core.llm.embedding_client import EmbeddingClient
from app.core.llm.sparse_embedding_client import SparseEmbeddingClient
from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.confidence_scorer import compute_confidence_breakdown
from app.rag_services.rag_service import RAGService
from app.rag_services.reranker import ReRankedChunk, ReRankOutcome
from app.rag_services.retrieval_strategy import (
    DenseRetrievalStrategy,
    HybridRetrievalStrategy,
    SparseRetrievalStrategy,
)
from app.repositories.semantic_cache_repository import SemanticQueryCache
from app.repositories.vector_repository import VectorRepository
from app.services.query_cache_service import QueryCacheService


class _InMemoryCacheBackend:
    """Minimal in-memory :class:`~app.services.query_cache_service.CacheBackend`."""

    def __init__(self, *, fail_get: bool = False, fail_set: bool = False) -> None:
        self._store: dict[str, str] = {}
        self._fail_get = fail_get
        self._fail_set = fail_set

    def get(self, key: str) -> str | None:
        if self._fail_get:
            raise ConnectionError("cache unavailable")
        return self._store.get(key)

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        if self._fail_set:
            raise ConnectionError("cache unavailable")
        self._store[key] = value

    def delete_prefix(self, prefix: str) -> int:
        keys = [k for k in self._store if k.startswith(prefix)]
        for k in keys:
            del self._store[k]
        return len(keys)


class _FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.1, 0.2] for _ in texts]


class _FakeVectorRepository:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self._results = results
        self.search_calls: list[dict[str, object]] = []

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        dense_embeddings: list[list[float]],
        sparse_embeddings: list[SparseVector],
    ) -> None:
        raise NotImplementedError

    def search_dense(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        self.search_calls.append({"query_embedding": query_embedding, "top_k": top_k})
        return self._results[:top_k]

    def search_sparse(self, query_sparse: SparseVector, top_k: int = 5) -> list[RetrievedChunk]:
        raise NotImplementedError

    def search_hybrid(
        self,
        query_embedding: list[float],
        query_sparse: SparseVector,
        top_k: int = 5,
        candidate_top_k: int = 20,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        self.search_calls.append({"query_embedding": query_embedding, "top_k": top_k})
        return self._results[:top_k]

    def scroll_all_chunks(self, limit: int = 10_000) -> list[dict[str, str]]:
        raise NotImplementedError


class _FakeSparseEmbeddingClient:
    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        raise NotImplementedError

    def embed_query(self, text: str) -> SparseVector:
        return SparseVector(indices=[1], values=[1.0])


class _FakeLLMClient:
    def __init__(self, answer: str = "the answer") -> None:
        self._answer = answer
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.calls.append({"system_prompt": system_prompt, "user_message": user_message})
        return LLMResponse(text=self._answer, usage=TokenUsage())

    def generate_json(
        self,
        system_prompt: str,
        user_message: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        raise NotImplementedError


class _FakeReranker:
    def __init__(self, *, name: str = "fake-reranker", fallback: bool = False) -> None:
        self._name = name
        self._fallback = fallback
        self.calls: list[dict[str, object]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def cache_namespace(self) -> str:
        return f"reranker:{self._name}:v1"

    def rerank(self, *, query: str, candidates, top_k: int) -> ReRankOutcome:
        self.calls.append({"query": query, "candidates": list(candidates), "top_k": top_k})
        # Reverses retrieval order so tests can tell reranking actually ran.
        reordered = list(reversed(candidates))[:top_k]
        items = tuple(
            ReRankedChunk(chunk=chunk, original_rank=rank, rerank_score=1.0)
            for rank, chunk in enumerate(reordered, start=1)
        )
        return ReRankOutcome(
            items=items,
            backend=self._name,
            applied=not self._fallback,
            fallback=self._fallback,
        )


class _FakeSemanticQueryCache:
    def __init__(
        self,
        *,
        match_key: str | None = None,
        raise_on_find: bool = False,
        raise_on_record: bool = False,
    ) -> None:
        self.match_key = match_key
        self._raise_on_find = raise_on_find
        self._raise_on_record = raise_on_record
        self.find_similar_calls: list[dict[str, object]] = []
        self.record_calls: list[dict[str, object]] = []

    def find_similar(
        self, *, query_embedding: list[float], cache_namespace: str, top_k: int
    ) -> str | None:
        self.find_similar_calls.append(
            {
                "query_embedding": query_embedding,
                "cache_namespace": cache_namespace,
                "top_k": top_k,
            }
        )
        if self._raise_on_find:
            raise RuntimeError("boom")
        return self.match_key

    def record(
        self, *, query_embedding: list[float], cache_namespace: str, top_k: int, cache_key: str
    ) -> None:
        self.record_calls.append(
            {
                "query_embedding": query_embedding,
                "cache_namespace": cache_namespace,
                "top_k": top_k,
                "cache_key": cache_key,
            }
        )
        if self._raise_on_record:
            raise RuntimeError("boom")


def _service(
    results: list[RetrievedChunk] | None = None,
    answer: str = "the answer",
    backend: _InMemoryCacheBackend | None = None,
) -> tuple[RAGService, _FakeEmbeddingClient, _FakeVectorRepository, _FakeLLMClient]:
    embedding_client = _FakeEmbeddingClient()
    vector_repository = _FakeVectorRepository(results or [])
    llm_client = _FakeLLMClient(answer)
    cache = QueryCacheService(backend or _InMemoryCacheBackend(), CacheSettings())
    dense_strategy = DenseRetrievalStrategy(
        vector_repository=cast(VectorRepository, vector_repository)
    )
    service = RAGService(
        embedding_client=cast(EmbeddingClient, embedding_client),
        retrieval_strategies={dense_strategy.name: dense_strategy},
        llm_client=cast(LLMClient, llm_client),
        cache=cache,
        default_retrieval_mode=dense_strategy.name,
    )
    return service, embedding_client, vector_repository, llm_client


def test_answer_returns_llm_text_and_sorted_unique_sources() -> None:
    chunks = [
        RetrievedChunk(text="a", source="b.pdf", score=0.9),
        RetrievedChunk(text="c", source="a.pdf", score=0.8),
        RetrievedChunk(text="d", source="a.pdf", score=0.7),
    ]
    service, _, _, _ = _service(results=chunks, answer="hello")

    response = service.answer("what is the policy?")

    assert response.answer == "hello"
    assert response.sources == ["a.pdf", "b.pdf"]
    assert response.cache_hit is False


def test_answer_confidence_is_computed_not_a_fixed_constant() -> None:
    strong_chunks = [
        RetrievedChunk(text="refunds within 30 days of purchase", source="a.pdf", score=0.9),
        RetrievedChunk(text="30 day refund window applies", source="b.pdf", score=0.88),
    ]
    weak_chunks = [RetrievedChunk(text="unrelated boilerplate", source="z.pdf", score=0.2)]

    strong_service, *_ = _service(
        results=strong_chunks, answer="Refunds are available within 30 days [a.pdf][b.pdf]."
    )
    weak_service, *_ = _service(results=weak_chunks, answer="I don't know.")

    strong_response = strong_service.answer("refund policy?")
    weak_response = weak_service.answer("refund policy?")

    assert strong_response.confidence != 0.7
    assert weak_response.confidence < strong_response.confidence


def test_answer_embeds_question_and_searches_with_top_k() -> None:
    service, embedding_client, vector_repository, _ = _service()

    service.answer("what is the policy?", top_k=3)

    assert embedding_client.calls == [["what is the policy?"]]
    assert vector_repository.search_calls[0]["top_k"] == 3


def test_sparse_mode_never_calls_the_dense_embedding_client() -> None:
    """Sparse retrieval must not incur OpenAI embedding cost/latency -
    RAGService.answer() should skip embed_texts() entirely when the
    selected strategy's requires_dense_embedding is False."""
    sparse_chunks = [RetrievedChunk(text="sparse hit", source="a.pdf", score=0.9)]

    class _SparseCapableRepo(_FakeVectorRepository):
        def search_sparse(self, query_sparse: SparseVector, top_k: int = 5) -> list[RetrievedChunk]:
            self.search_calls.append({"query_sparse": query_sparse, "top_k": top_k})
            return self._results[:top_k]

    sparse_repo = _SparseCapableRepo(sparse_chunks)
    embedding_client = _FakeEmbeddingClient()
    sparse_strategy = SparseRetrievalStrategy(
        vector_repository=cast(VectorRepository, sparse_repo),
        sparse_embedding_client=cast(SparseEmbeddingClient, _FakeSparseEmbeddingClient()),
    )
    service = RAGService(
        embedding_client=cast(EmbeddingClient, embedding_client),
        retrieval_strategies={sparse_strategy.name: sparse_strategy},
        llm_client=cast(LLMClient, _FakeLLMClient()),
        cache=QueryCacheService(_InMemoryCacheBackend(), CacheSettings()),
        default_retrieval_mode="sparse",
    )

    response = service.answer("refund policy?")

    assert embedding_client.calls == []
    assert response.sources == ["a.pdf"]


def test_hybrid_mode_confidence_uses_hybrid_calibration_not_dense() -> None:
    """RAGService.answer() must pass the selected strategy's name through
    to confidence scoring as retrieval_mode, not leave it defaulted to
    dense - otherwise a hybrid RRF-normalized score would be misjudged
    against the cosine-calibrated dense floor (see confidence_scorer's
    _evidence_coverage)."""
    low_rrf_chunks = [RetrievedChunk(text="hit", source="a.pdf", score=0.1)]
    repo = _FakeVectorRepository(low_rrf_chunks)
    hybrid_strategy = HybridRetrievalStrategy(
        vector_repository=cast(VectorRepository, repo),
        sparse_embedding_client=cast(SparseEmbeddingClient, _FakeSparseEmbeddingClient()),
    )
    service = RAGService(
        embedding_client=cast(EmbeddingClient, _FakeEmbeddingClient()),
        retrieval_strategies={hybrid_strategy.name: hybrid_strategy},
        llm_client=cast(LLMClient, _FakeLLMClient()),
        cache=QueryCacheService(_InMemoryCacheBackend(), CacheSettings()),
        default_retrieval_mode="hybrid",
    )

    response = service.answer("q")

    expected_hybrid = compute_confidence_breakdown(
        low_rrf_chunks, response.answer, retrieval_mode="hybrid"
    ).total
    expected_dense = compute_confidence_breakdown(
        low_rrf_chunks, response.answer, retrieval_mode="dense"
    ).total
    assert response.confidence == pytest.approx(expected_hybrid)
    assert response.confidence > expected_dense


def test_answer_builds_context_with_source_citations() -> None:
    chunks = [RetrievedChunk(text="refunds within 30 days", source="policy.pdf", score=0.9)]
    service, _, _, llm_client = _service(results=chunks)

    service.answer("what is the refund policy?")

    user_message = cast(str, llm_client.calls[0]["user_message"])
    assert "[policy.pdf]" in user_message
    assert "refunds within 30 days" in user_message
    assert "what is the refund policy?" in user_message


def test_answer_with_no_chunks_still_generates_an_answer() -> None:
    service, _, _, llm_client = _service(results=[])

    response = service.answer("anything?")

    assert response.sources == []
    assert response.metadata.retrieved_chunks == []
    assert "No relevant context" in cast(str, llm_client.calls[0]["user_message"])


def test_metadata_retrieved_chunks_mirror_search_results() -> None:
    chunks = [RetrievedChunk(text="hi", source="a.pdf", score=0.5, page_number=4)]
    service, _, _, _ = _service(results=chunks)

    response = service.answer("q")

    assert response.metadata.route == "rag"
    assert len(response.metadata.retrieved_chunks) == 1
    assert response.metadata.retrieved_chunks[0].text == "hi"
    assert response.metadata.retrieved_chunks[0].source == "a.pdf"
    assert response.metadata.retrieved_chunks[0].score == 0.5
    assert response.metadata.retrieved_chunks[0].page_number == 4


def test_second_call_with_same_args_is_a_cache_hit_and_skips_llm() -> None:
    chunks = [RetrievedChunk(text="hi", source="a.pdf", score=0.5)]
    service, embedding_client, _, llm_client = _service(results=chunks)

    first = service.answer("what is the policy?", top_k=5)
    second = service.answer("what is the policy?", top_k=5)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.answer == first.answer
    assert len(llm_client.calls) == 1  # second call was a cache hit
    assert len(embedding_client.calls) == 1


def test_different_top_k_is_not_a_cache_hit() -> None:
    service, _, _, llm_client = _service()

    service.answer("q", top_k=5)
    service.answer("q", top_k=10)

    assert len(llm_client.calls) == 2


def test_cache_read_failure_falls_back_to_computing_fresh() -> None:
    service, _, _, llm_client = _service(backend=_InMemoryCacheBackend(fail_get=True))

    response = service.answer("q")

    assert response.answer == "the answer"
    assert len(llm_client.calls) == 1


def test_cache_write_failure_does_not_crash() -> None:
    service, _, _, _ = _service(backend=_InMemoryCacheBackend(fail_set=True))

    response = service.answer("q")

    assert response.answer == "the answer"


def test_constructor_rejects_default_mode_absent_from_strategies() -> None:
    dense_strategy = DenseRetrievalStrategy(vector_repository=cast(VectorRepository, object()))

    with pytest.raises(ValueError, match="default_retrieval_mode"):
        RAGService(
            embedding_client=cast(EmbeddingClient, _FakeEmbeddingClient()),
            retrieval_strategies={dense_strategy.name: dense_strategy},
            llm_client=cast(LLMClient, _FakeLLMClient()),
            cache=QueryCacheService(_InMemoryCacheBackend(), CacheSettings()),
            default_retrieval_mode="hybrid",
        )


def test_retrieval_mode_override_selects_a_different_strategy_and_cache_bucket() -> None:
    dense_chunks = [RetrievedChunk(text="dense hit", source="a.pdf", score=0.9)]
    hybrid_chunks = [RetrievedChunk(text="hybrid hit", source="b.pdf", score=0.9)]
    dense_vector_repository = _FakeVectorRepository(dense_chunks)
    hybrid_vector_repository = _FakeVectorRepository(hybrid_chunks)
    dense_strategy = DenseRetrievalStrategy(
        vector_repository=cast(VectorRepository, dense_vector_repository)
    )
    hybrid_strategy = HybridRetrievalStrategy(
        vector_repository=cast(VectorRepository, hybrid_vector_repository),
        sparse_embedding_client=cast(SparseEmbeddingClient, _FakeSparseEmbeddingClient()),
    )
    service = RAGService(
        embedding_client=cast(EmbeddingClient, _FakeEmbeddingClient()),
        retrieval_strategies={dense_strategy.name: dense_strategy, hybrid_strategy.name: hybrid_strategy},
        llm_client=cast(LLMClient, _FakeLLMClient()),
        cache=QueryCacheService(_InMemoryCacheBackend(), CacheSettings()),
        default_retrieval_mode="dense",
    )

    default_response = service.answer("what is the policy?")
    hybrid_response = service.answer("what is the policy?", retrieval_mode="hybrid")

    assert default_response.metadata.retrieval_mode == "dense"
    assert default_response.sources == ["a.pdf"]
    assert hybrid_response.metadata.retrieval_mode == "hybrid"
    assert hybrid_response.sources == ["b.pdf"]
    # Distinct cache namespaces per mode - the hybrid call wasn't a cache hit
    # off the dense call's cached response.
    assert hybrid_response.cache_hit is False


def test_unknown_retrieval_mode_falls_back_to_default() -> None:
    service, _, _, _ = _service(results=[RetrievedChunk(text="x", source="a.pdf", score=0.9)])

    response = service.answer("q", retrieval_mode="nonexistent")

    assert response.metadata.retrieval_mode == "dense"


def _service_with_allowed_modes(
    allowed_retrieval_modes: frozenset[str] | None,
) -> RAGService:
    dense_repo = _FakeVectorRepository([RetrievedChunk(text="d", source="a.pdf", score=0.9)])
    hybrid_repo = _FakeVectorRepository([RetrievedChunk(text="h", source="b.pdf", score=0.9)])
    dense_strategy = DenseRetrievalStrategy(vector_repository=cast(VectorRepository, dense_repo))
    hybrid_strategy = HybridRetrievalStrategy(
        vector_repository=cast(VectorRepository, hybrid_repo),
        sparse_embedding_client=cast(SparseEmbeddingClient, _FakeSparseEmbeddingClient()),
    )
    return RAGService(
        embedding_client=cast(EmbeddingClient, _FakeEmbeddingClient()),
        retrieval_strategies={dense_strategy.name: dense_strategy, hybrid_strategy.name: hybrid_strategy},
        llm_client=cast(LLMClient, _FakeLLMClient()),
        cache=QueryCacheService(_InMemoryCacheBackend(), CacheSettings()),
        default_retrieval_mode="dense",
        allowed_retrieval_modes=allowed_retrieval_modes,
    )


def test_constructor_rejects_default_mode_not_in_allowed_modes() -> None:
    dense_strategy = DenseRetrievalStrategy(vector_repository=cast(VectorRepository, object()))

    with pytest.raises(ValueError, match="allowed_retrieval_modes"):
        RAGService(
            embedding_client=cast(EmbeddingClient, _FakeEmbeddingClient()),
            retrieval_strategies={dense_strategy.name: dense_strategy},
            llm_client=cast(LLMClient, _FakeLLMClient()),
            cache=QueryCacheService(_InMemoryCacheBackend(), CacheSettings()),
            default_retrieval_mode="dense",
            allowed_retrieval_modes=frozenset({"hybrid"}),
        )


def test_explicit_retrieval_mode_is_rejected_when_not_allowed() -> None:
    service = _service_with_allowed_modes(frozenset({"dense"}))

    with pytest.raises(HybridRetrievalDisabledError):
        service.answer("q", retrieval_mode="hybrid")


def test_dense_mode_is_always_allowed_even_when_hybrid_is_gated_off() -> None:
    service = _service_with_allowed_modes(frozenset({"dense"}))

    response = service.answer("q", retrieval_mode="dense")

    assert response.metadata.retrieval_mode == "dense"


def test_none_allowed_modes_permits_every_registered_strategy() -> None:
    """``allowed_retrieval_modes=None`` (the eval harness's bypass, see
    app.eval.invokers) permits every registered mode, regardless of any
    server-side rollout flag."""
    service = _service_with_allowed_modes(None)

    response = service.answer("q", retrieval_mode="hybrid")

    assert response.metadata.retrieval_mode == "hybrid"


def _service_with_reranker(
    results: list[RetrievedChunk],
    *,
    reranking_enabled: bool,
    reranker_initial_top_k: int = 20,
    reranker: _FakeReranker | None = None,
) -> tuple[RAGService, _FakeVectorRepository, _FakeReranker]:
    vector_repository = _FakeVectorRepository(results)
    reranker = reranker or _FakeReranker()
    dense_strategy = DenseRetrievalStrategy(
        vector_repository=cast(VectorRepository, vector_repository)
    )
    service = RAGService(
        embedding_client=cast(EmbeddingClient, _FakeEmbeddingClient()),
        retrieval_strategies={dense_strategy.name: dense_strategy},
        llm_client=cast(LLMClient, _FakeLLMClient()),
        cache=QueryCacheService(_InMemoryCacheBackend(), CacheSettings()),
        default_retrieval_mode=dense_strategy.name,
        reranker=reranker,
        reranker_initial_top_k=reranker_initial_top_k,
        reranking_enabled=reranking_enabled,
    )
    return service, vector_repository, reranker


def test_reranking_disabled_by_default_never_calls_the_reranker() -> None:
    chunks = [
        RetrievedChunk(text="a", source="a.pdf", score=0.9),
        RetrievedChunk(text="b", source="b.pdf", score=0.8),
    ]
    service, vector_repository, reranker = _service_with_reranker(
        chunks, reranking_enabled=False
    )

    service.answer("q", top_k=2)

    assert reranker.calls == []
    assert vector_repository.search_calls[0]["top_k"] == 2


def test_reranking_enabled_reorders_chunks_and_marks_metadata() -> None:
    chunks = [
        RetrievedChunk(text="first", source="a.pdf", score=0.9),
        RetrievedChunk(text="second", source="b.pdf", score=0.8),
    ]
    service, _, reranker = _service_with_reranker(chunks, reranking_enabled=True)

    response = service.answer("q", top_k=2)

    assert len(reranker.calls) == 1
    assert response.metadata.retrieved_chunks[0].text == "second"
    assert response.metadata.retrieved_chunks[1].text == "first"
    assert response.metadata.reranked is True


def test_reranking_expands_candidate_pool_to_reranker_initial_top_k() -> None:
    chunks = [RetrievedChunk(text=f"t{i}", source=f"s{i}.pdf", score=0.5) for i in range(10)]
    service, vector_repository, _ = _service_with_reranker(
        chunks, reranking_enabled=True, reranker_initial_top_k=8
    )

    service.answer("q", top_k=3)

    assert vector_repository.search_calls[0]["top_k"] == 8


def test_reranking_never_shrinks_candidate_pool_below_requested_top_k() -> None:
    chunks = [RetrievedChunk(text=f"t{i}", source=f"s{i}.pdf", score=0.5) for i in range(10)]
    service, vector_repository, _ = _service_with_reranker(
        chunks, reranking_enabled=True, reranker_initial_top_k=5
    )

    service.answer("q", top_k=8)

    assert vector_repository.search_calls[0]["top_k"] == 8


def test_reranking_skips_the_reranker_call_when_no_candidates_were_retrieved() -> None:
    service, _, reranker = _service_with_reranker([], reranking_enabled=True)

    response = service.answer("q")

    assert reranker.calls == []
    assert response.metadata.reranked is False


def test_default_response_metadata_reranked_is_false_without_a_reranker() -> None:
    service, _, _, _ = _service(results=[RetrievedChunk(text="x", source="a.pdf", score=0.9)])

    response = service.answer("q")

    assert response.metadata.reranked is False


def test_reranking_success_emits_a_debug_telemetry_log(caplog: pytest.LogCaptureFixture) -> None:
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    service, _, _ = _service_with_reranker(chunks, reranking_enabled=True)

    with caplog.at_level(logging.DEBUG, logger="app.rag_services.rag_service"):
        service.answer("q", top_k=1)

    records = [r for r in caplog.records if r.message == "rag.reranking_applied"]
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG
    assert records[0].backend == "fake-reranker"
    assert records[0].applied is True
    assert records[0].fallback is False
    assert records[0].candidate_count == 1
    assert records[0].result_count == 1
    assert not any(r.message == "rag.reranking_fallback" for r in caplog.records)


def test_reranking_fallback_emits_a_warning_not_a_debug_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    service, _, _ = _service_with_reranker(
        chunks, reranking_enabled=True, reranker=_FakeReranker(fallback=True)
    )

    with caplog.at_level(logging.DEBUG, logger="app.rag_services.rag_service"):
        service.answer("q", top_k=1)

    records = [r for r in caplog.records if r.message == "rag.reranking_fallback"]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert records[0].fallback is True
    assert not any(r.message == "rag.reranking_applied" for r in caplog.records)


def test_reranking_disabled_emits_no_reranking_logs(caplog: pytest.LogCaptureFixture) -> None:
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    service, _, _ = _service_with_reranker(chunks, reranking_enabled=False)

    with caplog.at_level(logging.DEBUG, logger="app.rag_services.rag_service"):
        service.answer("q", top_k=1)

    assert not any(r.message.startswith("rag.reranking") for r in caplog.records)


def _service_with_semantic_cache(
    results: list[RetrievedChunk],
    *,
    semantic_cache: _FakeSemanticQueryCache,
    semantic_cache_enabled: bool = True,
    llm_client: _FakeLLMClient | None = None,
) -> tuple[RAGService, _FakeLLMClient]:
    vector_repository = _FakeVectorRepository(results)
    llm_client = llm_client or _FakeLLMClient()
    dense_strategy = DenseRetrievalStrategy(
        vector_repository=cast(VectorRepository, vector_repository)
    )
    service = RAGService(
        embedding_client=cast(EmbeddingClient, _FakeEmbeddingClient()),
        retrieval_strategies={dense_strategy.name: dense_strategy},
        llm_client=cast(LLMClient, llm_client),
        cache=QueryCacheService(_InMemoryCacheBackend(), CacheSettings()),
        default_retrieval_mode=dense_strategy.name,
        semantic_cache=cast(SemanticQueryCache, semantic_cache),
        semantic_cache_enabled=semantic_cache_enabled,
    )
    return service, llm_client


def test_semantic_cache_disabled_by_default_never_calls_it() -> None:
    semantic_cache = _FakeSemanticQueryCache()
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    service, _ = _service_with_semantic_cache(
        chunks, semantic_cache=semantic_cache, semantic_cache_enabled=False
    )

    service.answer("q", top_k=3)

    assert semantic_cache.find_similar_calls == []
    assert semantic_cache.record_calls == []


def test_semantic_cache_miss_generates_normally_and_records_afterwards() -> None:
    semantic_cache = _FakeSemanticQueryCache(match_key=None)
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    service, llm_client = _service_with_semantic_cache(chunks, semantic_cache=semantic_cache)

    response = service.answer("what is the policy?", top_k=3)

    assert len(llm_client.calls) == 1
    assert response.cache_hit is False
    assert len(semantic_cache.find_similar_calls) == 1
    assert semantic_cache.find_similar_calls[0]["cache_namespace"] == "dense:v1"
    assert semantic_cache.find_similar_calls[0]["top_k"] == 3
    assert len(semantic_cache.record_calls) == 1
    assert semantic_cache.record_calls[0]["cache_namespace"] == "dense:v1"
    assert semantic_cache.record_calls[0]["top_k"] == 3


def test_semantic_cache_hit_serves_a_paraphrases_answer_without_regenerating() -> None:
    semantic_cache = _FakeSemanticQueryCache()
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    service, llm_client = _service_with_semantic_cache(
        chunks, semantic_cache=semantic_cache, llm_client=_FakeLLMClient("first answer")
    )

    first = service.answer("how is a loan classified after 91 days overdue?", top_k=3)
    assert len(llm_client.calls) == 1
    # Point the fake at the exact-match key the first call just recorded -
    # simulating a real semantic match against a differently-worded question.
    semantic_cache.match_key = semantic_cache.record_calls[0]["cache_key"]

    second = service.answer("once arrears pass 90 days, what classification applies?", top_k=3)

    assert len(llm_client.calls) == 1  # no second generation
    assert second.cache_hit is True
    assert second.answer == first.answer == "first answer"


def test_semantic_cache_match_pointing_at_an_expired_entry_falls_through() -> None:
    semantic_cache = _FakeSemanticQueryCache(match_key="a-key-that-was-never-cached")
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    service, llm_client = _service_with_semantic_cache(chunks, semantic_cache=semantic_cache)

    response = service.answer("q", top_k=3)

    assert len(llm_client.calls) == 1
    assert response.cache_hit is False


def test_semantic_cache_is_skipped_for_strategies_without_dense_embedding() -> None:
    """Sparse retrieval must not pay for an OpenAI embedding call just to
    feed the semantic cache - the same invariant that keeps it out of
    retrieval (test_sparse_mode_never_calls_the_dense_embedding_client)."""
    semantic_cache = _FakeSemanticQueryCache()
    sparse_chunks = [RetrievedChunk(text="sparse hit", source="a.pdf", score=0.9)]

    class _SparseCapableRepo(_FakeVectorRepository):
        def search_sparse(self, query_sparse: SparseVector, top_k: int = 5) -> list[RetrievedChunk]:
            self.search_calls.append({"query_sparse": query_sparse, "top_k": top_k})
            return self._results[:top_k]

    vector_repository = _SparseCapableRepo(sparse_chunks)
    sparse_strategy = SparseRetrievalStrategy(
        vector_repository=cast(VectorRepository, vector_repository),
        sparse_embedding_client=cast(SparseEmbeddingClient, _FakeSparseEmbeddingClient()),
    )
    service = RAGService(
        embedding_client=cast(EmbeddingClient, _FakeEmbeddingClient()),
        retrieval_strategies={sparse_strategy.name: sparse_strategy},
        llm_client=cast(LLMClient, _FakeLLMClient()),
        cache=QueryCacheService(_InMemoryCacheBackend(), CacheSettings()),
        default_retrieval_mode="sparse",
        semantic_cache=cast(SemanticQueryCache, semantic_cache),
        semantic_cache_enabled=True,
    )

    service.answer("q")

    assert semantic_cache.find_similar_calls == []
    assert semantic_cache.record_calls == []


def test_semantic_cache_lookup_failure_is_treated_as_a_miss(
    caplog: pytest.LogCaptureFixture,
) -> None:
    semantic_cache = _FakeSemanticQueryCache(raise_on_find=True)
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    service, llm_client = _service_with_semantic_cache(chunks, semantic_cache=semantic_cache)

    with caplog.at_level(logging.WARNING, logger="app.rag_services.rag_service"):
        response = service.answer("q", top_k=3)

    assert len(llm_client.calls) == 1
    assert response.cache_hit is False
    assert any(r.message == "rag.semantic_cache_read_failed" for r in caplog.records)


def test_semantic_cache_record_failure_does_not_crash_the_request(
    caplog: pytest.LogCaptureFixture,
) -> None:
    semantic_cache = _FakeSemanticQueryCache(raise_on_record=True)
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    service, llm_client = _service_with_semantic_cache(chunks, semantic_cache=semantic_cache)

    with caplog.at_level(logging.WARNING, logger="app.rag_services.rag_service"):
        response = service.answer("q", top_k=3)

    assert len(llm_client.calls) == 1
    assert response.answer == "the answer"
    assert any(r.message == "rag.semantic_cache_write_failed" for r in caplog.records)


def test_semantic_cache_partitions_by_reranking_cache_namespace() -> None:
    """A semantic hit must not leak across reranked vs. non-reranked
    answers for the same question - the namespace passed to the semantic
    cache must include the reranker suffix, same as the exact-match key."""
    semantic_cache = _FakeSemanticQueryCache()
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    vector_repository = _FakeVectorRepository(chunks)
    dense_strategy = DenseRetrievalStrategy(
        vector_repository=cast(VectorRepository, vector_repository)
    )
    service = RAGService(
        embedding_client=cast(EmbeddingClient, _FakeEmbeddingClient()),
        retrieval_strategies={dense_strategy.name: dense_strategy},
        llm_client=cast(LLMClient, _FakeLLMClient()),
        cache=QueryCacheService(_InMemoryCacheBackend(), CacheSettings()),
        default_retrieval_mode=dense_strategy.name,
        reranker=_FakeReranker(name="my-reranker"),
        reranking_enabled=True,
        semantic_cache=cast(SemanticQueryCache, semantic_cache),
        semantic_cache_enabled=True,
    )

    service.answer("q", top_k=3)

    assert semantic_cache.find_similar_calls[0]["cache_namespace"] == "dense:v1:reranker:my-reranker:v1"
    assert semantic_cache.record_calls[0]["cache_namespace"] == "dense:v1:reranker:my-reranker:v1"
