"""HyDE-specific integration tests for RAGService.answer().

Separate from test_rag_service.py (which predates HyDE and already covers
reranking/semantic-cache invariants at length) to keep each file focused -
these fakes are HyDE-specific (an embedding client that returns a
distinguishable vector for hypothesis text vs. the original question, so
tests can prove *which* vector actually reached retrieval/the semantic
cache).

Uses the real ``DynamicQueryTransformer`` (production adapter) wrapped
around a raw fake delegate, rather than a hand-rolled ``PlannedQueryTransformer``
fake - the private fallback config store RAGService builds when no
``config_store`` is passed seeds ``hyde_rollout_percentage=100``, which
makes ``DynamicQueryTransformer.plan`` deterministically resolve to the
"treatment" cohort for every query, so these tests don't need to think
about rollout sampling at all while still exercising the real
plan()/execute() contract (see test_dynamic_query_transformer.py for the
cohort-decision unit tests themselves).
"""

from __future__ import annotations

from typing import cast

from qdrant_client.models import SparseVector

from app.core.config.cache import CacheSettings
from app.core.llm.chat_client import LLMClient, LLMResponse, TokenUsage
from app.core.llm.embedding_client import EmbeddingClient
from app.core.llm.sparse_embedding_client import SparseEmbeddingClient
from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.dynamic_query_transformer import DynamicQueryTransformer
from app.rag_services.query_transformer import QueryTransformer, QueryTransformOutcome
from app.rag_services.rag_service import RAGService
from app.rag_services.reranker import PlannedReranker, ReRankedChunk, ReRankOutcome, RerankPlan
from app.rag_services.retrieval_strategy import (
    DenseRetrievalStrategy,
    HybridRetrievalStrategy,
    RetrievalStrategy,
    SparseRetrievalStrategy,
)
from app.repositories.semantic_cache_repository import SemanticQueryCache
from app.repositories.vector_repository import VectorRepository
from app.services.query_cache_service import QueryCacheService


class _InMemoryCacheBackend:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self._store[key] = value

    def delete_prefix(self, prefix: str) -> int:
        keys = [k for k in self._store if k.startswith(prefix)]
        for k in keys:
            del self._store[k]
        return len(keys)


class _FakeEmbeddingClient:
    """Returns [1.0, 0.0] for the original question and [0.0, 1.0] for any
    HyDE hypothesis text (identified by the "hyp" prefix the fakes below
    use), so tests can tell which vector reached retrieval/the semantic
    cache without inspecting call internals."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.0, 1.0] if t.startswith("hyp") else [1.0, 0.0] for t in texts]


class _FailingHyDEEmbeddingClient:
    """Embeds a single (original-question) text fine, but raises for a
    multi-text (HyDE hypothesis batch) call - simulates an embedding-stage
    failure specifically in the HyDE fusion path."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        self.calls.append(list(texts))
        if len(texts) > 1:
            raise RuntimeError("embedding batch failed")
        return [[1.0, 0.0] for _ in texts]


class _FakeVectorRepository:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self._results = results
        self.search_calls: list[dict[str, object]] = []

    def upsert_chunks(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError

    def search_dense(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        self.search_calls.append({"query_embedding": query_embedding, "top_k": top_k})
        return self._results[:top_k]

    def search_sparse(self, query_sparse: SparseVector, top_k: int = 5) -> list[RetrievedChunk]:
        self.search_calls.append({"query_sparse": query_sparse, "top_k": top_k})
        return self._results[:top_k]

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

    def delete_by_source(self, source: str) -> None:
        raise NotImplementedError


class _RecordingSparseEmbeddingClient:
    """Records the exact text handed to the sparse branch, so tests can
    prove hybrid retrieval's sparse side always gets the *original*
    question, never a HyDE hypothesis."""

    def __init__(self) -> None:
        self.embed_query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        raise NotImplementedError

    def embed_query(self, text: str) -> SparseVector:
        self.embed_query_calls.append(text)
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

    def generate_json(self, *args: object, **kwargs: object) -> LLMResponse:
        raise NotImplementedError


class _FakeReranker:
    """PlannedReranker fake, always "treatment" when enabled - see the
    identical pattern in test_rag_service.py."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def plan(self, query: str, config, *, enabled: bool) -> RerankPlan:
        if not enabled:
            return RerankPlan("disabled", None, "none", "disabled", "rerank:none:reason=disabled")
        return RerankPlan("treatment", None, "fake-reranker", None, "reranker:fake-reranker:v1")

    def execute(
        self, plan: RerankPlan, *, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> ReRankOutcome:
        self.calls.append({"query": query, "top_k": top_k})
        items = tuple(
            ReRankedChunk(chunk=c, original_rank=i + 1, rerank_score=1.0)
            for i, c in enumerate(candidates[:top_k])
        )
        return ReRankOutcome(items=items, backend="fake-reranker", applied=True)


class _FakeSemanticQueryCache:
    def __init__(self, *, candidate_keys: list[str] | None = None) -> None:
        self.candidate_keys = candidate_keys or []
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


class _FakeQueryTransformer:
    """Two hypotheses by default (text prefixed "hyp" so
    _FakeEmbeddingClient can recognize them); pass ``outcome_factory`` or
    ``raise_error`` to simulate bypass/fallback/error outcomes. Implements
    the *delegate*-level QueryTransformer Protocol (name/cache_namespace/
    transform) - _hyde_service wraps this in the real
    DynamicQueryTransformer, same as production."""

    def __init__(self, *, raise_error: bool = False, outcome_factory=None) -> None:  # type: ignore[no-untyped-def]
        self._raise_error = raise_error
        self._outcome_factory = outcome_factory
        self.calls: list[str] = []
        self.cache_namespace_value = "hyde:fake:v1"

    @property
    def name(self) -> str:
        return "fake-hyde"

    @property
    def cache_namespace(self) -> str:
        return self.cache_namespace_value

    def transform(self, query: str) -> QueryTransformOutcome:
        self.calls.append(query)
        if self._raise_error:
            raise RuntimeError("boom")
        if self._outcome_factory is not None:
            return self._outcome_factory(query)
        return QueryTransformOutcome(
            retrieval_texts=("hyp one", "hyp two"), backend="fake-hyde", applied=True
        )


def _hyde_service(
    results: list[RetrievedChunk],
    embedding_client: object,
    query_transformer: object,
    *,
    hyde_enabled: bool = True,
    llm_client: _FakeLLMClient | None = None,
    reranker: PlannedReranker | None = None,
    reranking_enabled: bool = False,
    semantic_cache: _FakeSemanticQueryCache | None = None,
    semantic_cache_enabled: bool = False,
    strategy_name: str = "dense",
    sparse_embedding_client: object = None,
) -> tuple[RAGService, _FakeVectorRepository]:
    vector_repository = _FakeVectorRepository(results)
    sparse_client = sparse_embedding_client or _RecordingSparseEmbeddingClient()
    strategy: RetrievalStrategy
    if strategy_name == "hybrid":
        strategy = HybridRetrievalStrategy(
            vector_repository=cast(VectorRepository, vector_repository),
            sparse_embedding_client=cast(SparseEmbeddingClient, sparse_client),
        )
    elif strategy_name == "sparse":
        strategy = SparseRetrievalStrategy(
            vector_repository=cast(VectorRepository, vector_repository),
            sparse_embedding_client=cast(SparseEmbeddingClient, sparse_client),
        )
    else:
        strategy = DenseRetrievalStrategy(
            vector_repository=cast(VectorRepository, vector_repository)
        )
    service = RAGService(
        embedding_client=cast(EmbeddingClient, embedding_client),
        retrieval_strategies={strategy.name: strategy},
        llm_client=cast(LLMClient, llm_client or _FakeLLMClient()),
        cache=QueryCacheService(_InMemoryCacheBackend(), CacheSettings()),
        default_retrieval_mode=strategy.name,
        reranker=reranker,
        reranking_enabled=reranking_enabled,
        semantic_cache=cast(SemanticQueryCache, semantic_cache) if semantic_cache else None,
        semantic_cache_enabled=semantic_cache_enabled,
        # hyde_rollout_percentage=100 is the private fallback store's
        # default (see RAGService.__init__) - DynamicQueryTransformer.plan
        # therefore always resolves "treatment" here whenever hyde_enabled
        # is true, with no rollout sampling for these tests to reason about.
        query_transformer=DynamicQueryTransformer(
            delegate=cast(QueryTransformer, query_transformer)
        ),
        hyde_enabled=hyde_enabled,
    )
    return service, vector_repository


def test_exact_cache_hit_skips_embedding_and_hyde() -> None:
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    embedding_client = _FakeEmbeddingClient()
    transformer = _FakeQueryTransformer()
    service, _ = _hyde_service(chunks, embedding_client, transformer)

    first = service.answer("q", top_k=1)
    embed_calls_after_first = len(embedding_client.calls)
    transform_calls_after_first = len(transformer.calls)
    second = service.answer("q", top_k=1)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(embedding_client.calls) == embed_calls_after_first
    assert len(transformer.calls) == transform_calls_after_first


def test_semantic_cache_hit_uses_original_vector_and_skips_hyde() -> None:
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    embedding_client = _FakeEmbeddingClient()
    transformer = _FakeQueryTransformer()
    semantic_cache = _FakeSemanticQueryCache()
    service, _ = _hyde_service(
        chunks,
        embedding_client,
        transformer,
        semantic_cache=semantic_cache,
        semantic_cache_enabled=True,
    )

    service.answer("q1", top_k=1)
    assert len(transformer.calls) == 1
    semantic_cache.candidate_keys = [cast(str, semantic_cache.record_calls[0]["cache_key"])]

    second = service.answer("q2", top_k=1)

    assert second.cache_hit is True
    assert len(transformer.calls) == 1  # unchanged - the semantic hit never reached HyDE
    assert semantic_cache.find_candidates_calls[0]["query_embedding"] == [1.0, 0.0]
    assert semantic_cache.record_calls[0]["query_embedding"] == [1.0, 0.0]


def test_dense_hyde_success_batch_embeds_only_the_hypotheses() -> None:
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    embedding_client = _FakeEmbeddingClient()
    transformer = _FakeQueryTransformer()
    service, vector_repository = _hyde_service(chunks, embedding_client, transformer)

    service.answer("what is the wire transfer limit?", top_k=1)

    assert embedding_client.calls == [["hyp one", "hyp two"]]
    assert vector_repository.search_calls[0]["query_embedding"] == [0.0, 1.0]


def test_hybrid_retrieval_uses_fused_hyde_vector_and_original_sparse_text() -> None:
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    embedding_client = _FakeEmbeddingClient()
    transformer = _FakeQueryTransformer()
    sparse_client = _RecordingSparseEmbeddingClient()
    service, vector_repository = _hyde_service(
        chunks,
        embedding_client,
        transformer,
        strategy_name="hybrid",
        sparse_embedding_client=sparse_client,
    )

    service.answer("original question text", top_k=1)

    assert vector_repository.search_calls[0]["query_embedding"] == [0.0, 1.0]
    assert sparse_client.embed_query_calls == ["original question text"]


def test_sparse_only_never_calls_hyde_or_the_dense_embedding_client() -> None:
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    embedding_client = _FakeEmbeddingClient()
    transformer = _FakeQueryTransformer()
    service, _ = _hyde_service(chunks, embedding_client, transformer, strategy_name="sparse")

    service.answer("q", top_k=1)

    assert embedding_client.calls == []
    assert transformer.calls == []


def test_reranker_receives_the_original_question_not_a_hypothesis() -> None:
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    embedding_client = _FakeEmbeddingClient()
    transformer = _FakeQueryTransformer()
    reranker = _FakeReranker()
    service, _ = _hyde_service(
        chunks, embedding_client, transformer, reranker=reranker, reranking_enabled=True
    )

    service.answer("what is the refund window?", top_k=1)

    assert reranker.calls[0]["query"] == "what is the refund window?"


def test_answer_prompt_never_contains_hypothetical_passage_text() -> None:
    chunks = [RetrievedChunk(text="refunds within 30 days", source="policy.pdf", score=0.9)]
    embedding_client = _FakeEmbeddingClient()
    transformer = _FakeQueryTransformer()
    llm_client = _FakeLLMClient()
    service, _ = _hyde_service(chunks, embedding_client, transformer, llm_client=llm_client)

    service.answer("what is the refund policy?", top_k=1)

    user_message = cast(str, llm_client.calls[0]["user_message"])
    assert "what is the refund policy?" in user_message
    assert "refunds within 30 days" in user_message
    assert "hyp one" not in user_message
    assert "hyp two" not in user_message


def test_hyde_transform_failure_falls_back_to_original_vector_and_does_not_cache() -> None:
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    embedding_client = _FakeEmbeddingClient()
    raising = _FakeQueryTransformer(raise_error=True)
    # DynamicQueryTransformer wraps its delegate in FailOpenQueryTransformer
    # internally (see _hyde_service) - no need to pre-wrap here.
    service, vector_repository = _hyde_service(chunks, embedding_client, raising)

    first = service.answer("q", top_k=1)
    second = service.answer("q", top_k=1)

    assert vector_repository.search_calls[0]["query_embedding"] == [1.0, 0.0]
    assert first.metadata.hyde.fallback is True
    assert first.cache_hit is False
    assert second.cache_hit is False


def test_hyde_embedding_or_fusion_failure_falls_back_to_original_vector_and_does_not_cache() -> (
    None
):
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    embedding_client = _FailingHyDEEmbeddingClient()
    transformer = (
        _FakeQueryTransformer()
    )  # applied=True, 2 hypotheses -> the batch embed call fails
    service, vector_repository = _hyde_service(chunks, embedding_client, transformer)

    first = service.answer("q", top_k=1)
    second = service.answer("q", top_k=1)

    assert vector_repository.search_calls[0]["query_embedding"] == [1.0, 0.0]
    assert first.metadata.hyde.applied is False
    assert first.metadata.hyde.fallback is True
    assert first.metadata.hyde.bypass_reason == "embedding_or_fusion_error"
    assert first.cache_hit is False
    assert second.cache_hit is False


def test_hyde_bypass_without_fallback_is_still_cached() -> None:
    """A rollout bypass (applied=False, fallback=False - the delegate was
    never even invoked) is not degraded output, so unlike a real fallback
    it's safe to cache under its own versioned namespace."""
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    embedding_client = _FakeEmbeddingClient()
    transformer = _FakeQueryTransformer(
        outcome_factory=lambda q: QueryTransformOutcome(
            retrieval_texts=(q,), backend="none", applied=False, bypass_reason="rollout"
        )
    )
    service, _ = _hyde_service(chunks, embedding_client, transformer)

    first = service.answer("q", top_k=1)
    second = service.answer("q", top_k=1)

    assert first.metadata.hyde.applied is False
    assert first.metadata.hyde.fallback is False
    assert first.cache_hit is False
    assert second.cache_hit is True


def test_cache_key_changes_when_the_hyde_transformer_namespace_changes() -> None:
    """Same reasoning as the reranker candidate-pool-size cache-key test in
    test_rag_service.py: model/prompt/N/rollout/emergency state all fold
    into cache_namespace via the transformer's own cache_namespace."""
    service_a, _ = _hyde_service([], _FakeEmbeddingClient(), _FakeQueryTransformer())

    other = _FakeQueryTransformer()
    other.cache_namespace_value = "hyde:fake:v2"
    service_b, _ = _hyde_service([], _FakeEmbeddingClient(), other)

    key_a = service_a._cache_key("q", 1, "dense:v1:hyde:fake:v1")
    key_b = service_b._cache_key("q", 1, "dense:v1:hyde:fake:v2")

    assert key_a != key_b


def test_semantic_cache_record_uses_original_vector_even_when_hyde_applied() -> None:
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    embedding_client = _FakeEmbeddingClient()
    transformer = _FakeQueryTransformer()
    semantic_cache = _FakeSemanticQueryCache()
    service, vector_repository = _hyde_service(
        chunks,
        embedding_client,
        transformer,
        semantic_cache=semantic_cache,
        semantic_cache_enabled=True,
    )

    service.answer("q", top_k=1)

    assert vector_repository.search_calls[0]["query_embedding"] == [0.0, 1.0]
    assert semantic_cache.record_calls[0]["query_embedding"] == [1.0, 0.0]


def test_hyde_metadata_disabled_when_hyde_enabled_is_false() -> None:
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    transformer = _FakeQueryTransformer()
    service, _ = _hyde_service(chunks, _FakeEmbeddingClient(), transformer, hyde_enabled=False)

    response = service.answer("q", top_k=1)

    assert response.metadata.hyde.enabled is False
    assert response.metadata.hyde.applied is False
    assert response.metadata.hyde.backend == "none"
    assert transformer.calls == []


def test_hyde_metadata_applied_on_success() -> None:
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    service, _ = _hyde_service(chunks, _FakeEmbeddingClient(), _FakeQueryTransformer())

    response = service.answer("q", top_k=1)

    assert response.metadata.hyde.enabled is True
    assert response.metadata.hyde.applied is True
    assert response.metadata.hyde.fallback is False
    assert response.metadata.hyde.hypothesis_count == 2
    assert response.metadata.hyde.backend == "fake-hyde"


def test_hyde_metadata_fallback_on_transform_error() -> None:
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    raising = _FakeQueryTransformer(raise_error=True)
    service, _ = _hyde_service(chunks, _FakeEmbeddingClient(), raising)

    response = service.answer("q", top_k=1)

    assert response.metadata.hyde.enabled is True
    assert response.metadata.hyde.applied is False
    assert response.metadata.hyde.fallback is True


def test_hyde_metadata_rollout_bypass_is_distinct_from_fallback() -> None:
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    transformer = _FakeQueryTransformer(
        outcome_factory=lambda q: QueryTransformOutcome(
            retrieval_texts=(q,), backend="none", applied=False, bypass_reason="rollout"
        )
    )
    service, _ = _hyde_service(chunks, _FakeEmbeddingClient(), transformer)

    response = service.answer("q", top_k=1)

    assert response.metadata.hyde.enabled is True
    assert response.metadata.hyde.applied is False
    assert response.metadata.hyde.fallback is False
    assert response.metadata.hyde.bypass_reason == "rollout"


class _FakeMetricsRecorder:
    def __init__(self) -> None:
        self.hyde_attempt_calls: list[dict[str, object]] = []
        self.hyde_bypass_calls: list[str] = []

    def record_semantic_cache_lookup(self, *, hit: bool) -> None:
        pass

    def record_hyde_attempt(self, *, duration_ms: float, fallback: bool, usage_tokens: int) -> None:
        self.hyde_attempt_calls.append(
            {"duration_ms": duration_ms, "fallback": fallback, "usage_tokens": usage_tokens}
        )

    def record_hyde_bypass(self, *, reason: str) -> None:
        self.hyde_bypass_calls.append(reason)


def test_treatment_attempt_metrics_include_total_stage_latency_not_just_generation() -> None:
    """Regression: the old code recorded the metric right after the LLM
    generation call returned, before the embedding+fusion stage could still
    fail - HydeQueryTransformer itself never sets duration_ms, so a metric
    recorded that early would always read ~0ms and could report "success"
    for a request that later failed during embedding/fusion. RAGService
    must record exactly once, after the complete stage, with a real
    measured duration."""
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    metrics = _FakeMetricsRecorder()
    embedding_client = _FakeEmbeddingClient()
    transformer = _FakeQueryTransformer()
    vector_repository = _FakeVectorRepository(chunks)
    dense_strategy = DenseRetrievalStrategy(
        vector_repository=cast(VectorRepository, vector_repository)
    )
    service = RAGService(
        embedding_client=cast(EmbeddingClient, embedding_client),
        retrieval_strategies={dense_strategy.name: dense_strategy},
        llm_client=cast(LLMClient, _FakeLLMClient()),
        cache=QueryCacheService(_InMemoryCacheBackend(), CacheSettings()),
        default_retrieval_mode=dense_strategy.name,
        query_transformer=DynamicQueryTransformer(delegate=cast(QueryTransformer, transformer)),
        hyde_enabled=True,
        metrics=cast("object", metrics),
    )

    service.answer("q", top_k=1)

    assert len(metrics.hyde_attempt_calls) == 1
    assert metrics.hyde_attempt_calls[0]["fallback"] is False
    assert isinstance(metrics.hyde_attempt_calls[0]["duration_ms"], float)


def test_embedding_fusion_failure_is_recorded_as_a_fallback_attempt_not_a_success() -> None:
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    metrics = _FakeMetricsRecorder()
    embedding_client = _FailingHyDEEmbeddingClient()
    transformer = _FakeQueryTransformer()
    vector_repository = _FakeVectorRepository(chunks)
    dense_strategy = DenseRetrievalStrategy(
        vector_repository=cast(VectorRepository, vector_repository)
    )
    service = RAGService(
        embedding_client=cast(EmbeddingClient, embedding_client),
        retrieval_strategies={dense_strategy.name: dense_strategy},
        llm_client=cast(LLMClient, _FakeLLMClient()),
        cache=QueryCacheService(_InMemoryCacheBackend(), CacheSettings()),
        default_retrieval_mode=dense_strategy.name,
        query_transformer=DynamicQueryTransformer(delegate=cast(QueryTransformer, transformer)),
        hyde_enabled=True,
        metrics=cast("object", metrics),
    )

    service.answer("q", top_k=1)

    assert len(metrics.hyde_attempt_calls) == 1
    assert metrics.hyde_attempt_calls[0]["fallback"] is True


def test_metrics_none_does_not_crash_when_hyde_and_semantic_cache_are_active() -> None:
    """Mirrors the eval harness, which constructs RAGService without a
    metrics= kwarg at all - RAGService must default to a no-op recorder,
    not require every caller to supply one."""
    chunks = [RetrievedChunk(text="a", source="a.pdf", score=0.9)]
    semantic_cache = _FakeSemanticQueryCache()
    service, _ = _hyde_service(
        chunks,
        _FakeEmbeddingClient(),
        _FakeQueryTransformer(),
        semantic_cache=semantic_cache,
        semantic_cache_enabled=True,
    )

    response = service.answer("q", top_k=1)

    assert response.metadata.hyde.applied is True
