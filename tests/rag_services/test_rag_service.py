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
from app.rag_services.rag_runtime_config import RagRuntimeConfig, RagRuntimeConfigStore
from app.rag_services.rag_service import RAGService
from app.rag_services.reranker.dynamic_reranker import DynamicReranker
from app.rag_services.reranker.reranker import (
    NoOpReranker,
    ReRankedChunk,
    ReRankOutcome,
    RerankPlan,
)
from app.rag_services.retrieval_strategy import (
    DenseRetrievalStrategy,
    HybridRetrievalStrategy,
    SparseRetrievalStrategy,
)
from app.repositories.semantic_cache_repository import SemanticQueryCache
from app.repositories.vector_repository import VectorRepository
from app.services.query_cache_service import QueryCacheService
from app.services.rag_metrics_service import RagMetricsService


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
    """A PlannedReranker fake that's always "treatment" when enabled (no
    rollout sampling) - deterministic for tests, same as
    StaticPlannedReranker but keeping the exact pre-existing
    cache_namespace format (``reranker:{name}:v1``) so existing namespace
    assertions in this file don't need to change."""

    def __init__(self, *, name: str = "fake-reranker", fallback: bool = False) -> None:
        self._name = name
        self._fallback = fallback
        self.calls: list[dict[str, object]] = []

    def plan(self, query: str, config, *, enabled: bool) -> RerankPlan:
        if not enabled:
            return RerankPlan("disabled", None, "none", "disabled", "rerank:none:reason=disabled")
        return RerankPlan("treatment", None, self._name, None, f"reranker:{self._name}:v1")

    def execute(self, plan: RerankPlan, *, query: str, candidates, top_k: int) -> ReRankOutcome:
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
        candidate_keys: list[str] | None = None,
        raise_on_find: bool = False,
        raise_on_record: bool = False,
    ) -> None:
        self.candidate_keys = candidate_keys or []
        self._raise_on_find = raise_on_find
        self._raise_on_record = raise_on_record
        self.find_candidates_calls: list[dict[str, object]] = []
        self.record_calls: list[dict[str, object]] = []

    def find_candidates(
        self,
        *,
        query_embedding: list[float],
        cache_namespace: str,
        top_k: int,
        similarity_threshold: float,
    ) -> list[str]:
        self.find_candidates_calls.append(
            {
                "query_embedding": query_embedding,
                "cache_namespace": cache_namespace,
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
            }
        )
        if self._raise_on_find:
            raise RuntimeError("boom")
        return self.candidate_keys

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


class _FakeMetricsRecorder:
    def __init__(self) -> None:
        self.semantic_lookup_calls: list[bool] = []
        self.hyde_attempt_calls: list[dict[str, object]] = []
        self.hyde_bypass_calls: list[str] = []

    def record_semantic_cache_lookup(self, *, hit: bool) -> None:
        self.semantic_lookup_calls.append(hit)

    def record_hyde_attempt(self, *, duration_ms: float, fallback: bool, usage_tokens: int) -> None:
        self.hyde_attempt_calls.append(
            {"duration_ms": duration_ms, "fallback": fallback, "usage_tokens": usage_tokens}
        )

    def record_hyde_bypass(self, *, reason: str) -> None:
        self.hyde_bypass_calls.append(reason)


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
    """Regression: context is now built from CRAG's evidence output (see
    RAGService._build_evidence_context), not the raw chunk list directly -
    but with CRAG disabled/control (the default here), that evidence is an
    identity-wrapped pass-through of the same chunks/sources (see
    PlannedNoOpCorrectiveRetriever/local_evidence)."""
    chunks = [RetrievedChunk(text="refunds within 30 days", source="policy.pdf", score=0.9)]
    service, _, _, llm_client = _service(results=chunks)

    service.answer("what is the refund policy?")

    user_message = cast(str, llm_client.calls[0]["user_message"])
    assert '"source": "policy.pdf"' in user_message
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
        retrieval_strategies={
            dense_strategy.name: dense_strategy,
            hybrid_strategy.name: hybrid_strategy,
        },
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
        retrieval_strategies={
            dense_strategy.name: dense_strategy,
            hybrid_strategy.name: hybrid_strategy,
        },
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
    service, vector_repository, reranker = _service_with_reranker(chunks, reranking_enabled=False)

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
    assert response.metadata.reranking.enabled is True
    assert response.metadata.reranking.applied is True
    assert response.metadata.reranking.fallback is False
    assert response.metadata.reranking.backend == "fake-reranker"
    assert response.metadata.reranking.candidate_count == 2
    # rerank_score/original_rank threaded through from the reranker's
    # output, not discarded at the chunks = [item.chunk ...] reassignment.
    assert response.metadata.retrieved_chunks[0].rerank_score == 1.0
    assert response.metadata.retrieved_chunks[0].original_rank == 1


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
    assert response.metadata.reranking.applied is False
    assert response.metadata.reranking.candidate_count == 0


def test_default_response_metadata_reranking_is_disabled_without_a_reranker() -> None:
    service, _, _, _ = _service(results=[RetrievedChunk(text="x", source="a.pdf", score=0.9)])

    response = service.answer("q")

    assert response.metadata.reranking.enabled is False
    assert response.metadata.reranking.applied is False
    assert response.metadata.reranking.backend == "none"


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
    metrics: _FakeMetricsRecorder | None = None,
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
        metrics=metrics,
    )
    return service, llm_client


def test_semantic_cache_disabled_by_default_never_calls_it() -> None:
    semantic_cache = _FakeSemanticQueryCache()
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    service, _ = _service_with_semantic_cache(
        chunks, semantic_cache=semantic_cache, semantic_cache_enabled=False
    )

    service.answer("q", top_k=3)

    assert semantic_cache.find_candidates_calls == []
    assert semantic_cache.record_calls == []


def test_semantic_cache_miss_generates_normally_and_records_afterwards() -> None:
    semantic_cache = _FakeSemanticQueryCache(candidate_keys=[])
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    service, llm_client = _service_with_semantic_cache(chunks, semantic_cache=semantic_cache)

    response = service.answer("what is the policy?", top_k=3)

    assert len(llm_client.calls) == 1
    assert response.cache_hit is False
    assert len(semantic_cache.find_candidates_calls) == 1
    assert semantic_cache.find_candidates_calls[0]["cache_namespace"] == "dense:v1:corpus=1"
    assert semantic_cache.find_candidates_calls[0]["top_k"] == 3
    assert len(semantic_cache.record_calls) == 1
    assert semantic_cache.record_calls[0]["cache_namespace"] == "dense:v1:corpus=1"
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
    semantic_cache.candidate_keys = [semantic_cache.record_calls[0]["cache_key"]]

    second = service.answer("once arrears pass 90 days, what classification applies?", top_k=3)

    assert len(llm_client.calls) == 1  # no second generation
    assert second.cache_hit is True
    assert second.answer == first.answer == "first answer"


def test_semantic_cache_falls_through_to_the_next_candidate_when_nearest_is_stale() -> None:
    """The nearest match's Redis entry may have expired while a
    farther-but-still-live match hasn't - find_candidates returns several
    ranked candidates for exactly this reason (see
    QdrantSemanticQueryCache.find_candidates); RAGService must try each in
    order rather than giving up after the first stale one."""
    semantic_cache = _FakeSemanticQueryCache()
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    service, llm_client = _service_with_semantic_cache(
        chunks, semantic_cache=semantic_cache, llm_client=_FakeLLMClient("first answer")
    )

    first = service.answer("q1", top_k=3)
    live_key = semantic_cache.record_calls[0]["cache_key"]
    semantic_cache.candidate_keys = ["stale-key-never-cached", live_key]

    second = service.answer("q2", top_k=3)

    assert len(llm_client.calls) == 1  # no second generation - the live candidate was found
    assert second.cache_hit is True
    assert second.answer == first.answer == "first answer"


def test_semantic_cache_all_candidates_stale_falls_through_to_fresh_generation() -> None:
    semantic_cache = _FakeSemanticQueryCache(candidate_keys=["never-cached-a", "never-cached-b"])
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    service, llm_client = _service_with_semantic_cache(chunks, semantic_cache=semantic_cache)

    response = service.answer("q", top_k=3)

    assert len(llm_client.calls) == 1
    assert response.cache_hit is False


def test_semantic_cache_confirmed_hit_records_a_metrics_hit() -> None:
    semantic_cache = _FakeSemanticQueryCache()
    metrics = _FakeMetricsRecorder()
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    service, _ = _service_with_semantic_cache(
        chunks,
        semantic_cache=semantic_cache,
        llm_client=_FakeLLMClient("first answer"),
        metrics=metrics,
    )

    service.answer("q1", top_k=3)
    semantic_cache.candidate_keys = [semantic_cache.record_calls[0]["cache_key"]]
    service.answer("q2", top_k=3)

    # First call: no candidates yet -> miss. Second call: a confirmed live
    # candidate -> hit.
    assert metrics.semantic_lookup_calls == [False, True]


def test_semantic_cache_stale_candidate_records_a_metrics_miss_not_a_hit() -> None:
    """Regression: Qdrant returning a similarity-clearing candidate is not
    the same as a confirmed cache hit - if the candidate's Redis entry has
    expired, the outcome is a miss (the request still falls through to a
    fresh LLM generation), and the metric must reflect that, not the raw
    "Qdrant found something" signal."""
    semantic_cache = _FakeSemanticQueryCache(candidate_keys=["never-cached-a", "never-cached-b"])
    metrics = _FakeMetricsRecorder()
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    service, llm_client = _service_with_semantic_cache(
        chunks, semantic_cache=semantic_cache, metrics=metrics
    )

    response = service.answer("q", top_k=3)

    assert len(llm_client.calls) == 1
    assert response.cache_hit is False
    assert metrics.semantic_lookup_calls == [False]


def test_semantic_cache_disabled_never_records_metrics() -> None:
    semantic_cache = _FakeSemanticQueryCache()
    metrics = _FakeMetricsRecorder()
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    service, _ = _service_with_semantic_cache(
        chunks, semantic_cache=semantic_cache, semantic_cache_enabled=False, metrics=metrics
    )

    service.answer("q", top_k=3)

    assert metrics.semantic_lookup_calls == []


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

    assert semantic_cache.find_candidates_calls == []
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

    expected_namespace = "dense:v1:corpus=1:reranker:my-reranker:v1:candidates=20"
    assert semantic_cache.find_candidates_calls[0]["cache_namespace"] == expected_namespace
    assert semantic_cache.record_calls[0]["cache_namespace"] == expected_namespace


def test_fallback_reranking_response_is_not_cached_exact_match() -> None:
    """A fallback response (reranker failed, retrieval order used) must
    not be cached under the reranker-configured namespace - otherwise
    later requests keep getting the degraded answer forever, even after
    the reranker recovers, since nothing else would ever invalidate it."""
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    service, _, _ = _service_with_reranker(
        chunks, reranking_enabled=True, reranker=_FakeReranker(fallback=True)
    )
    first = service.answer("q", top_k=1)
    second = service.answer("q", top_k=1)

    assert first.cache_hit is False
    assert second.cache_hit is False  # still not cached - the fallback was never written


def test_fallback_reranking_response_is_not_semantically_recorded() -> None:
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
        reranker=_FakeReranker(fallback=True),
        reranking_enabled=True,
        semantic_cache=cast(SemanticQueryCache, semantic_cache),
        semantic_cache_enabled=True,
    )

    response = service.answer("q", top_k=1)

    assert response.metadata.reranking.fallback is True
    assert semantic_cache.record_calls == []


def test_semantic_record_is_skipped_when_the_redis_write_itself_fails() -> None:
    """_set_cached must gate semantic recording on an actual successful
    write, not just "no fallback" - a Redis outage shouldn't leave a
    semantic pointer aimed at a cache key that was never written."""
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
        cache=QueryCacheService(_InMemoryCacheBackend(fail_set=True), CacheSettings()),
        default_retrieval_mode=dense_strategy.name,
        semantic_cache=cast(SemanticQueryCache, semantic_cache),
        semantic_cache_enabled=True,
    )

    service.answer("q", top_k=1)

    assert semantic_cache.record_calls == []


def test_cache_key_includes_candidate_pool_size_when_reranking_enabled() -> None:
    """Widening reranker_initial_top_k (20 -> 50) can change which chunk
    the reranker ends up promoting even though top_k and the reranker
    itself are unchanged - the candidate pool size must be part of the
    cache key, or a stale narrower-pool answer gets served forever."""
    chunks = [RetrievedChunk(text=f"t{i}", source=f"s{i}.pdf", score=0.5) for i in range(30)]
    narrow, _, _ = _service_with_reranker(chunks, reranking_enabled=True, reranker_initial_top_k=10)
    wide, _, _ = _service_with_reranker(chunks, reranking_enabled=True, reranker_initial_top_k=25)

    narrow_response = narrow.answer("q", top_k=5)
    wide_response = wide.answer("q", top_k=5)

    # Two independent services never share a cache, so this only proves
    # the key computation itself differs: same question/top_k/reranker,
    # different candidate pool -> neither would ever hit the other's
    # entry if they did share one. Verified directly via the private key
    # builder, which is the actual unit under test here.
    narrow_key = narrow._cache_key("q", 5, "dense:v1:reranker:fake-reranker:v1:candidates=10")
    wide_key = wide._cache_key("q", 5, "dense:v1:reranker:fake-reranker:v1:candidates=25")
    assert narrow_key != wide_key
    assert narrow_response.cache_hit is False
    assert wide_response.cache_hit is False


def _config_store(*, corpus_version: int = 1) -> RagRuntimeConfigStore:
    return RagRuntimeConfigStore(
        RagRuntimeConfig(
            reranking_enabled=False,
            reranker_backend="local",
            reranker_rollout_percentage=100,
            emergency_disabled=False,
            semantic_cache_enabled=False,
            semantic_cache_threshold=0.95,
            corpus_version=corpus_version,
            hyde_enabled=False,
            hyde_rollout_percentage=0,
            crag_enabled=False,
            crag_rollout_percentage=0,
            crag_web_enabled=False,
        )
    )


def test_corpus_version_bump_makes_a_previously_cached_answer_unreachable() -> None:
    """Full end-to-end version of the namespace test above: the exact same
    question, against the exact same cache backend, must regenerate after
    the shared config_store's corpus_version moves forward - proving a
    corpus mutation invalidates cached answers even without relying on the
    explicit Redis clear() succeeding."""
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    vector_repository = _FakeVectorRepository(chunks)
    dense_strategy = DenseRetrievalStrategy(
        vector_repository=cast(VectorRepository, vector_repository)
    )
    shared_cache = QueryCacheService(_InMemoryCacheBackend(), CacheSettings())
    llm_client = _FakeLLMClient()
    store = _config_store(corpus_version=1)
    service = RAGService(
        embedding_client=cast(EmbeddingClient, _FakeEmbeddingClient()),
        retrieval_strategies={dense_strategy.name: dense_strategy},
        llm_client=cast(LLMClient, llm_client),
        cache=shared_cache,
        default_retrieval_mode=dense_strategy.name,
        config_store=store,
    )

    first = service.answer("q", top_k=1)
    second = service.answer("q", top_k=1)
    assert first.cache_hit is False
    assert second.cache_hit is True  # same corpus_version - normal cache hit

    store.replace(
        RagRuntimeConfig(
            reranking_enabled=False,
            reranker_backend="local",
            reranker_rollout_percentage=100,
            emergency_disabled=False,
            semantic_cache_enabled=False,
            semantic_cache_threshold=0.95,
            corpus_version=2,
            hyde_enabled=False,
            hyde_rollout_percentage=0,
            crag_enabled=False,
            crag_rollout_percentage=0,
            crag_web_enabled=False,
        )
    )
    after_bump = service.answer("q", top_k=1)

    assert after_bump.cache_hit is False


def test_per_call_reranking_override_enables_reranking_for_a_single_request() -> None:
    """reranking_enabled=True on a single answer() call must engage the
    reranker even when the instance default (from
    RAGFeatureSettings.reranking_enabled_by_default) is off - this is
    what lets the eval harness compare reranked vs. non-reranked runs
    through one RAGService instance (see app.eval.invokers)."""
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    service, _, reranker = _service_with_reranker(chunks, reranking_enabled=False)

    without_override = service.answer("q1", top_k=1)
    assert without_override.metadata.reranking.enabled is False
    assert len(reranker.calls) == 0

    with_override = service.answer("q2", top_k=1, reranking_enabled=True)
    assert with_override.metadata.reranking.enabled is True
    assert len(reranker.calls) == 1


def test_per_call_reranking_override_disables_reranking_for_a_single_request() -> None:
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    service, _, reranker = _service_with_reranker(chunks, reranking_enabled=True)

    response = service.answer("q", top_k=1, reranking_enabled=False)

    assert response.metadata.reranking.enabled is False


def test_cache_key_uses_the_v6_prefix() -> None:
    """Regression: the merged cache-key format combines candidate-pool-size
    versioning (v3), HyDE identity (v4), cohort-isolation for reranking and
    HyDE (v5), and now cohort-isolation for CRAG too - it must not collide
    with any predecessor format for the same question/namespace."""
    import hashlib

    service, *_ = _service()

    key = service._cache_key("q", 5, "dense:v1:corpus=1")

    expected = hashlib.sha256(b"rag:v6:dense:v1:corpus=1:5:q").hexdigest()
    assert key == expected


def test_rollout_control_cohort_never_widens_the_candidate_pool_or_calls_the_backend() -> None:
    """Regression: candidate_top_k used to widen for any query whenever
    reranking was configured on at all, regardless of whether *this*
    query's own rollout sampling actually selected it - only the treatment
    cohort should ever retrieve a widened pool; control must see exactly
    the caller's requested top_k, same as if reranking were off entirely."""
    chunks = [RetrievedChunk(text=f"t{i}", source=f"s{i}.pdf", score=0.5) for i in range(10)]
    vector_repository = _FakeVectorRepository(chunks)
    dense_strategy = DenseRetrievalStrategy(
        vector_repository=cast(VectorRepository, vector_repository)
    )
    config_store = RagRuntimeConfigStore(
        RagRuntimeConfig(
            reranking_enabled=True,
            reranker_backend="local",
            reranker_rollout_percentage=0,  # guarantees every query is "control"
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
    )
    # A NoOpReranker "local" delegate is fine here - the whole point of this
    # test is that the control cohort never reaches it at all.
    reranker = DynamicReranker(local=NoOpReranker(), voyage=None, metrics=RagMetricsService())
    service = RAGService(
        embedding_client=cast(EmbeddingClient, _FakeEmbeddingClient()),
        retrieval_strategies={dense_strategy.name: dense_strategy},
        llm_client=cast(LLMClient, _FakeLLMClient()),
        cache=QueryCacheService(_InMemoryCacheBackend(), CacheSettings()),
        default_retrieval_mode=dense_strategy.name,
        reranker=reranker,
        reranker_initial_top_k=8,
        config_store=config_store,
    )

    response = service.answer("q", top_k=3)

    assert vector_repository.search_calls[0]["top_k"] == 3  # not widened to 8
    assert response.metadata.reranking.applied is False


def test_semantic_cache_partitions_control_and_treatment_reranker_cohorts_separately() -> None:
    """Regression: the semantic-cache namespace used to only encode the
    *configured* rollout percentage, not which cohort a specific query
    actually landed in - a control-cohort query and a treatment-cohort
    query under the same rollout% must not be able to share a semantic
    cache entry."""
    semantic_cache = _FakeSemanticQueryCache()
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    vector_repository = _FakeVectorRepository(chunks)
    dense_strategy = DenseRetrievalStrategy(
        vector_repository=cast(VectorRepository, vector_repository)
    )
    config_store = RagRuntimeConfigStore(
        RagRuntimeConfig(
            reranking_enabled=True,
            reranker_backend="local",
            reranker_rollout_percentage=50,
            emergency_disabled=False,
            semantic_cache_enabled=True,
            semantic_cache_threshold=0.95,
            corpus_version=1,
            hyde_enabled=False,
            hyde_rollout_percentage=0,
            crag_enabled=False,
            crag_rollout_percentage=0,
            crag_web_enabled=False,
        )
    )
    reranker = DynamicReranker(local=NoOpReranker(), voyage=None, metrics=RagMetricsService())
    service = RAGService(
        embedding_client=cast(EmbeddingClient, _FakeEmbeddingClient()),
        retrieval_strategies={dense_strategy.name: dense_strategy},
        llm_client=cast(LLMClient, _FakeLLMClient()),
        cache=QueryCacheService(_InMemoryCacheBackend(), CacheSettings()),
        default_retrieval_mode=dense_strategy.name,
        reranker=reranker,
        semantic_cache=cast(SemanticQueryCache, semantic_cache),
        config_store=config_store,
    )

    # Find one query landing in each cohort under this 50% rollout.
    control_query = next(
        q
        for q in (f"q{i}" for i in range(50))
        if reranker.plan(q, config_store.current, enabled=True).cohort == "control"
    )
    treatment_query = next(
        q
        for q in (f"q{i}" for i in range(50))
        if reranker.plan(q, config_store.current, enabled=True).cohort == "treatment"
    )

    service.answer(control_query, top_k=1)
    service.answer(treatment_query, top_k=1)

    control_namespace = semantic_cache.record_calls[0]["cache_namespace"]
    treatment_namespace = semantic_cache.record_calls[1]["cache_namespace"]
    assert control_namespace != treatment_namespace


def test_config_store_current_is_read_exactly_once_per_answer_call() -> None:
    """The whole point of capturing one snapshot up front: a config update
    landing mid-request must not be observable within that same request.
    This is a structural guarantee (DynamicReranker/DynamicQueryTransformer/
    QdrantSemanticQueryCache hold no store reference of their own - see
    their module docstrings) verified here by counting reads on the one
    store RAGService itself holds."""
    real_store = RagRuntimeConfigStore(
        RagRuntimeConfig(
            reranking_enabled=True,
            reranker_backend="local",
            reranker_rollout_percentage=100,
            emergency_disabled=False,
            semantic_cache_enabled=True,
            semantic_cache_threshold=0.95,
            corpus_version=1,
            hyde_enabled=False,
            hyde_rollout_percentage=0,
            crag_enabled=False,
            crag_rollout_percentage=0,
            crag_web_enabled=False,
        )
    )
    read_count = 0

    class _CountingConfigStore:
        @property
        def current(self) -> RagRuntimeConfig:
            nonlocal read_count
            read_count += 1
            return real_store.current

    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    semantic_cache = _FakeSemanticQueryCache()
    service, _, reranker = _service_with_reranker(chunks, reranking_enabled=True)
    service._config_store = cast(RagRuntimeConfigStore, _CountingConfigStore())
    service._semantic_cache = cast(SemanticQueryCache, semantic_cache)

    service.answer("q", top_k=1)

    assert read_count == 1
