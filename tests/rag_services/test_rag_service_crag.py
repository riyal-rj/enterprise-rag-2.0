"""CRAG-specific integration tests for RAGService.answer().

Mirrors test_rag_service_hyde.py's shape: uses the real
``DynamicCorrectiveRetriever`` (production adapter) wrapped around a raw
fake delegate, rather than a hand-rolled ``PlannedCorrectiveRetriever`` fake
- the private fallback config store RAGService builds when no
``config_store`` is passed seeds ``crag_rollout_percentage=100``, so
``DynamicCorrectiveRetriever.plan`` deterministically resolves "treatment"
for every query whenever ``crag_enabled=True``, letting these tests ignore
rollout sampling entirely (see test_dynamic_corrective_retriever.py for the
cohort-decision unit tests themselves).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from qdrant_client.models import SparseVector

import app.rag_services.rag_service as rag_service_module
from app.core.config.cache import CacheSettings
from app.core.llm.chat_client import LLMClient, LLMResponse, TokenUsage
from app.core.llm.embedding_client import EmbeddingClient
from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.crag import (
    CorrectiveRetriever,
    CRAGDecision,
    CRAGOutcome,
    EvidenceChunk,
    EvidenceOrigin,
    local_evidence,
)
from app.rag_services.crag.dynamic_corrective_retriever import DynamicCorrectiveRetriever
from app.rag_services.rag_runtime_config import RagRuntimeConfig, RagRuntimeConfigStore
from app.rag_services.rag_service import RAGService
from app.rag_services.reranker import PlannedReranker, ReRankedChunk, ReRankOutcome, RerankPlan
from app.rag_services.retrieval_strategy import DenseRetrievalStrategy, RetrievalStrategy
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
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.0, 1.0] if t.startswith("hyp") else [1.0, 0.0] for t in texts]


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
        raise NotImplementedError

    def search_hybrid(
        self,
        query_embedding: list[float],
        query_sparse: SparseVector,
        top_k: int = 5,
        candidate_top_k: int = 20,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        raise NotImplementedError

    def scroll_all_chunks(self, limit: int = 10_000) -> list[dict[str, str]]:
        raise NotImplementedError

    def delete_by_source(self, source: str) -> None:
        raise NotImplementedError


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
    """Reverses candidate order and reports applied=True - so tests can
    distinguish "CRAG saw retrieval order" from "CRAG saw post-rerank
    order"."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def plan(self, query: str, config: object, *, enabled: bool) -> RerankPlan:
        if not enabled:
            return RerankPlan("disabled", None, "none", "disabled", "rerank:none:reason=disabled")
        return RerankPlan("treatment", None, "fake-reranker", None, "reranker:fake-reranker:v1")

    def execute(
        self, plan: RerankPlan, *, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> ReRankOutcome:
        self.calls.append({"query": query, "top_k": top_k})
        reversed_candidates = list(reversed(candidates))[:top_k]
        items = tuple(
            ReRankedChunk(chunk=c, original_rank=len(candidates) - i, rerank_score=1.0)
            for i, c in enumerate(reversed_candidates)
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
        self.find_candidates_calls.append({"cache_namespace": cache_namespace})
        return self.candidate_keys

    def record(
        self, *, query_embedding: list[float], cache_namespace: str, top_k: int, cache_key: str
    ) -> None:
        self.record_calls.append({"cache_key": cache_key})


class _FakeCorrectiveDelegate:
    """Delegate-level CorrectiveRetriever fake - _crag_service wraps this in
    the real DynamicCorrectiveRetriever, same as production."""

    def __init__(self, *, outcome_factory=None, raise_error: bool = False) -> None:  # type: ignore[no-untyped-def]
        self._outcome_factory = outcome_factory
        self._raise_error = raise_error
        self.calls: list[dict[str, object]] = []

    @property
    def cache_namespace(self) -> str:
        return "crag:fake:v1"

    def correct(
        self, question: str, chunks: list[RetrievedChunk], *, allow_web: bool
    ) -> CRAGOutcome:
        self.calls.append({"question": question, "chunks": list(chunks), "allow_web": allow_web})
        if self._raise_error:
            raise RuntimeError("boom")
        if self._outcome_factory is not None:
            return self._outcome_factory(question, chunks)
        return CRAGOutcome(
            evidence=local_evidence(chunks), decision=CRAGDecision.CORRECT, applied=True
        )


def _crag_service(
    results: list[RetrievedChunk],
    corrective_delegate: object,
    *,
    crag_enabled: bool = True,
    llm_client: _FakeLLMClient | None = None,
    reranker: PlannedReranker | None = None,
    reranking_enabled: bool = False,
    semantic_cache: _FakeSemanticQueryCache | None = None,
    semantic_cache_enabled: bool = False,
    metrics: object = None,
    config_store: RagRuntimeConfigStore | None = None,
) -> tuple[RAGService, _FakeVectorRepository]:
    vector_repository = _FakeVectorRepository(results)
    strategy: RetrievalStrategy = DenseRetrievalStrategy(
        vector_repository=cast(VectorRepository, vector_repository)
    )
    kwargs: dict[str, object] = dict(
        embedding_client=cast(EmbeddingClient, _FakeEmbeddingClient()),
        retrieval_strategies={strategy.name: strategy},
        llm_client=cast(LLMClient, llm_client or _FakeLLMClient()),
        cache=QueryCacheService(_InMemoryCacheBackend(), CacheSettings()),
        default_retrieval_mode=strategy.name,
        reranker=reranker,
        reranking_enabled=reranking_enabled,
        semantic_cache=cast(SemanticQueryCache, semantic_cache) if semantic_cache else None,
        semantic_cache_enabled=semantic_cache_enabled,
        corrective_retriever=DynamicCorrectiveRetriever(
            delegate=cast(CorrectiveRetriever, corrective_delegate)
        ),
        crag_enabled=crag_enabled,
    )
    if metrics is not None:
        kwargs["metrics"] = metrics
    if config_store is not None:
        kwargs["config_store"] = config_store
    service = RAGService(**kwargs)  # type: ignore[arg-type]
    return service, vector_repository


def _chunks(*texts: str) -> list[RetrievedChunk]:
    return [RetrievedChunk(text=t, source=f"{t}.pdf", score=0.9) for t in texts]


def test_crag_sees_post_rerank_order_not_retrieval_order() -> None:
    chunks = _chunks("first", "second", "third")
    delegate = _FakeCorrectiveDelegate()
    reranker = _FakeReranker()
    service, _ = _crag_service(chunks, delegate, reranker=reranker, reranking_enabled=True)

    service.answer("q", top_k=3)

    seen_texts = [c.text for c in cast(list, delegate.calls[0]["chunks"])]
    assert seen_texts == ["third", "second", "first"]


def test_hyde_hypotheses_never_reach_crag() -> None:
    """CRAG always receives the original question text, never a HyDE
    hypothesis - execute() always passes RAGService.answer()'s own
    ``question`` variable, which HyDE never mutates."""
    chunks = _chunks("a")
    delegate = _FakeCorrectiveDelegate()
    service, _ = _crag_service(chunks, delegate)

    service.answer("what is the wire transfer limit?", top_k=1)

    assert delegate.calls[0]["question"] == "what is the wire transfer limit?"


def test_exact_cache_hit_bypasses_crag() -> None:
    chunks = _chunks("a")
    delegate = _FakeCorrectiveDelegate()
    service, _ = _crag_service(chunks, delegate)

    first = service.answer("q", top_k=1)
    calls_after_first = len(delegate.calls)
    second = service.answer("q", top_k=1)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(delegate.calls) == calls_after_first


def test_semantic_cache_hit_bypasses_crag() -> None:
    chunks = _chunks("a")
    delegate = _FakeCorrectiveDelegate()
    semantic_cache = _FakeSemanticQueryCache()
    service, _ = _crag_service(
        chunks, delegate, semantic_cache=semantic_cache, semantic_cache_enabled=True
    )

    service.answer("q1", top_k=1)
    assert len(delegate.calls) == 1
    semantic_cache.candidate_keys = [cast(str, semantic_cache.record_calls[0]["cache_key"])]

    second = service.answer("q2", top_k=1)

    assert second.cache_hit is True
    assert len(delegate.calls) == 1  # unchanged - the semantic hit never reached CRAG


def test_final_answer_context_equals_final_evidence() -> None:
    chunks = _chunks("a")

    def factory(question: str, chunks: list[RetrievedChunk]) -> CRAGOutcome:
        return CRAGOutcome(
            evidence=(
                EvidenceChunk(
                    text="policy sentence",
                    source="policy.pdf",
                    page_number=3,
                    retrieval_score=0.9,
                    origin=EvidenceOrigin.POLICY,
                ),
                EvidenceChunk(
                    text="web sentence",
                    source="RBI Circular",
                    page_number=None,
                    retrieval_score=0.8,
                    origin=EvidenceOrigin.REGULATORY_WEB,
                    canonical_url="https://rbi.org.in/x",
                    retrieved_at_iso="2026-01-01T00:00:00+00:00",
                ),
            ),
            decision=CRAGDecision.CORRECT,
            applied=True,
            web_used=True,
        )

    delegate = _FakeCorrectiveDelegate(outcome_factory=factory)
    llm_client = _FakeLLMClient()
    service, _ = _crag_service(chunks, delegate, llm_client=llm_client)

    response = service.answer("q", top_k=1)

    assert [e.text for e in response.metadata.final_evidence] == ["policy sentence", "web sentence"]
    user_message = cast(str, llm_client.calls[0]["user_message"])
    assert "policy sentence" in user_message
    assert "web sentence" in user_message


def test_confidence_uses_final_evidence_not_raw_retrieved_chunks() -> None:
    chunks = _chunks("raw chunk text")

    def factory(question: str, chunks: list[RetrievedChunk]) -> CRAGOutcome:
        return CRAGOutcome(
            evidence=(
                EvidenceChunk(
                    text="corrected evidence text",
                    source="a.pdf",
                    page_number=None,
                    retrieval_score=0.9,
                    origin=EvidenceOrigin.POLICY,
                ),
            ),
            decision=CRAGDecision.CORRECT,
            applied=True,
        )

    delegate = _FakeCorrectiveDelegate(outcome_factory=factory)
    service, _ = _crag_service(chunks, delegate)

    captured: dict[str, object] = {}
    real = rag_service_module.compute_confidence_breakdown

    def spy(chunks_arg, *args, **kwargs):  # type: ignore[no-untyped-def]
        captured["chunks"] = chunks_arg
        return real(chunks_arg, *args, **kwargs)

    rag_service_module.compute_confidence_breakdown = spy  # type: ignore[assignment]
    try:
        service.answer("q", top_k=1)
    finally:
        rag_service_module.compute_confidence_breakdown = real  # type: ignore[assignment]

    seen_texts = [c.text for c in cast(list, captured["chunks"])]
    assert seen_texts == ["corrected evidence text"]
    assert "raw chunk text" not in seen_texts


def test_claim_checking_uses_final_evidence_not_raw_retrieved_chunks() -> None:
    chunks = _chunks("raw chunk text")

    def factory(question: str, chunks: list[RetrievedChunk]) -> CRAGOutcome:
        return CRAGOutcome(
            evidence=(
                EvidenceChunk(
                    text="corrected evidence text",
                    source="a.pdf",
                    page_number=None,
                    retrieval_score=0.9,
                    origin=EvidenceOrigin.POLICY,
                ),
            ),
            decision=CRAGDecision.CORRECT,
            applied=True,
        )

    delegate = _FakeCorrectiveDelegate(outcome_factory=factory)
    service, _ = _crag_service(chunks, delegate)

    captured: dict[str, object] = {}
    real = rag_service_module.find_unsupported_claims

    def spy(answer_text, chunks_arg):  # type: ignore[no-untyped-def]
        captured["chunks"] = chunks_arg
        return real(answer_text, chunks_arg)

    rag_service_module.find_unsupported_claims = spy  # type: ignore[assignment]
    try:
        service.answer("q", top_k=1)
    finally:
        rag_service_module.find_unsupported_claims = real  # type: ignore[assignment]

    seen_texts = [c.text for c in cast(list, captured["chunks"])]
    assert seen_texts == ["corrected evidence text"]


def test_fallback_response_is_not_cached() -> None:
    chunks = _chunks("a")

    def factory(question: str, chunks: list[RetrievedChunk]) -> CRAGOutcome:
        return CRAGOutcome(
            evidence=local_evidence(chunks), decision=None, applied=False, fallback=True
        )

    delegate = _FakeCorrectiveDelegate(outcome_factory=factory)
    service, _ = _crag_service(chunks, delegate)

    first = service.answer("q", top_k=1)
    second = service.answer("q", top_k=1)

    assert first.cache_hit is False
    assert second.cache_hit is False
    assert len(delegate.calls) == 2


def test_abstention_response_is_not_cached() -> None:
    chunks = _chunks("a")

    def factory(question: str, chunks: list[RetrievedChunk]) -> CRAGOutcome:
        return CRAGOutcome(
            evidence=local_evidence(chunks),
            decision=CRAGDecision.INCORRECT,
            applied=True,
            abstain=True,
        )

    delegate = _FakeCorrectiveDelegate(outcome_factory=factory)
    service, _ = _crag_service(chunks, delegate)

    first = service.answer("q", top_k=1)
    second = service.answer("q", top_k=1)

    assert first.cache_hit is False
    assert second.cache_hit is False
    assert len(delegate.calls) == 2


def test_web_augmented_answer_is_not_cached_even_when_not_abstaining() -> None:
    chunks = _chunks("a")

    def factory(question: str, chunks: list[RetrievedChunk]) -> CRAGOutcome:
        return CRAGOutcome(
            evidence=local_evidence(chunks),
            decision=CRAGDecision.CORRECT,
            applied=True,
            web_used=True,
            abstain=False,
        )

    delegate = _FakeCorrectiveDelegate(outcome_factory=factory)
    service, _ = _crag_service(chunks, delegate)

    first = service.answer("q", top_k=1)
    second = service.answer("q", top_k=1)

    assert first.cache_hit is False
    assert second.cache_hit is False
    assert len(delegate.calls) == 2


def test_control_and_treatment_cohorts_use_isolated_caches() -> None:
    chunks = _chunks("a")
    delegate = _FakeCorrectiveDelegate()
    config_store = RagRuntimeConfigStore(
        RagRuntimeConfig(
            reranking_enabled=False,
            reranker_backend="local",
            reranker_rollout_percentage=100,
            emergency_disabled=False,
            semantic_cache_enabled=False,
            semantic_cache_threshold=0.95,
            corpus_version=1,
            hyde_enabled=False,
            hyde_rollout_percentage=0,
            crag_enabled=True,
            crag_rollout_percentage=50,
            crag_web_enabled=False,
        )
    )
    service, _ = _crag_service(chunks, delegate, config_store=config_store)

    control_question = next(
        q
        for q in (f"q{i}" for i in range(50))
        if service._corrective_retriever.plan(q, config_store.current, enabled=True).cohort
        == "control"
    )
    treatment_question = next(
        q
        for q in (f"q{i}" for i in range(50))
        if service._corrective_retriever.plan(q, config_store.current, enabled=True).cohort
        == "treatment"
    )

    control_key = service._cache_key(
        control_question,
        1,
        service._corrective_retriever.plan(
            control_question, config_store.current, enabled=True
        ).cache_namespace,
    )
    treatment_key = service._cache_key(
        treatment_question,
        1,
        service._corrective_retriever.plan(
            treatment_question, config_store.current, enabled=True
        ).cache_namespace,
    )

    assert control_key != treatment_key


class _CountingConfigStore(RagRuntimeConfigStore):
    def __init__(self, initial: RagRuntimeConfig) -> None:
        super().__init__(initial)
        self.read_count = 0

    @property
    def current(self) -> RagRuntimeConfig:  # type: ignore[override]
        self.read_count += 1
        return super().current


def test_answer_reads_the_config_store_exactly_once_per_request() -> None:
    chunks = _chunks("a")
    delegate = _FakeCorrectiveDelegate()
    config_store = _CountingConfigStore(
        RagRuntimeConfig(
            reranking_enabled=False,
            reranker_backend="local",
            reranker_rollout_percentage=100,
            emergency_disabled=False,
            semantic_cache_enabled=False,
            semantic_cache_threshold=0.95,
            corpus_version=1,
            hyde_enabled=False,
            hyde_rollout_percentage=0,
            crag_enabled=True,
            crag_rollout_percentage=100,
            crag_web_enabled=False,
        )
    )
    service, _ = _crag_service(chunks, delegate, config_store=config_store)
    config_store.read_count = 0  # reset after construction-time reads, if any

    service.answer("q", top_k=1)

    assert config_store.read_count == 1


def test_structured_canonical_web_citation_is_returned_in_sources() -> None:
    chunks = _chunks("a")

    def factory(question: str, chunks: list[RetrievedChunk]) -> CRAGOutcome:
        return CRAGOutcome(
            evidence=(
                EvidenceChunk(
                    text="web sentence",
                    source="RBI Circular on KYC",
                    page_number=None,
                    retrieval_score=0.8,
                    origin=EvidenceOrigin.REGULATORY_WEB,
                    canonical_url="https://rbi.org.in/circular/123",
                    retrieved_at_iso="2026-01-01T00:00:00+00:00",
                ),
            ),
            decision=CRAGDecision.CORRECT,
            applied=True,
            web_used=True,
        )

    delegate = _FakeCorrectiveDelegate(outcome_factory=factory)
    service, _ = _crag_service(chunks, delegate)

    response = service.answer("q", top_k=1)

    assert response.sources == ["https://rbi.org.in/circular/123"]
    assert response.metadata.final_evidence[0].canonical_url == "https://rbi.org.in/circular/123"


def test_shadow_mode_runs_crag_but_never_serves_its_evidence() -> None:
    chunks = _chunks("a")

    def factory(question: str, chunks: list[RetrievedChunk]) -> CRAGOutcome:
        return CRAGOutcome(
            evidence=(
                EvidenceChunk(
                    text="shadow-only corrected text",
                    source="a.pdf",
                    page_number=None,
                    retrieval_score=0.9,
                    origin=EvidenceOrigin.POLICY,
                ),
            ),
            decision=CRAGDecision.CORRECT,
            applied=True,
        )

    delegate = _FakeCorrectiveDelegate(outcome_factory=factory)
    llm_client = _FakeLLMClient()
    config_store = RagRuntimeConfigStore(
        RagRuntimeConfig(
            reranking_enabled=False,
            reranker_backend="local",
            reranker_rollout_percentage=100,
            emergency_disabled=False,
            semantic_cache_enabled=False,
            semantic_cache_threshold=0.95,
            corpus_version=1,
            hyde_enabled=False,
            hyde_rollout_percentage=0,
            crag_enabled=True,
            crag_rollout_percentage=100,
            crag_web_enabled=False,
            crag_shadow_enabled=True,
        )
    )
    service, _ = _crag_service(chunks, delegate, llm_client=llm_client, config_store=config_store)

    response = service.answer("q", top_k=1)

    # The delegate was actually invoked (observation) ...
    assert len(delegate.calls) == 1
    # ... but its evidence never reached the answer LLM or the response.
    user_message = cast(str, llm_client.calls[0]["user_message"])
    assert "shadow-only corrected text" not in user_message
    assert [e.text for e in response.metadata.final_evidence] == ["a"]
    assert response.metadata.crag.bypass_reason == "shadow_not_served"
    assert response.metadata.crag.applied is False


class _FakeCRAGTelemetry:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def record_attempt(
        self,
        *,
        cohort: str,
        decision: str | None,
        fallback: bool,
        abstain: bool,
        web_used: bool,
        duration_ms: float,
        usage_tokens: int,
        served: bool,
    ) -> None:
        self.calls.append(
            {
                "cohort": cohort,
                "decision": decision,
                "fallback": fallback,
                "abstain": abstain,
                "web_used": web_used,
                "served": served,
            }
        )


def test_fleet_telemetry_records_a_served_treatment_attempt() -> None:
    chunks = _chunks("a")
    delegate = _FakeCorrectiveDelegate()
    telemetry = _FakeCRAGTelemetry()
    vector_repository = _FakeVectorRepository(chunks)
    strategy = DenseRetrievalStrategy(vector_repository=cast(VectorRepository, vector_repository))
    service = RAGService(
        embedding_client=cast(EmbeddingClient, _FakeEmbeddingClient()),
        retrieval_strategies={strategy.name: strategy},
        llm_client=cast(LLMClient, _FakeLLMClient()),
        cache=QueryCacheService(_InMemoryCacheBackend(), CacheSettings()),
        default_retrieval_mode=strategy.name,
        corrective_retriever=DynamicCorrectiveRetriever(
            delegate=cast(CorrectiveRetriever, delegate)
        ),
        crag_enabled=True,
        crag_telemetry=telemetry,  # type: ignore[arg-type]
    )

    service.answer("q", top_k=1)

    assert len(telemetry.calls) == 1
    assert telemetry.calls[0]["cohort"] == "treatment"
    assert telemetry.calls[0]["served"] is True


def test_fleet_telemetry_records_an_unserved_shadow_attempt() -> None:
    chunks = _chunks("a")
    delegate = _FakeCorrectiveDelegate()
    telemetry = _FakeCRAGTelemetry()
    config_store = RagRuntimeConfigStore(
        RagRuntimeConfig(
            reranking_enabled=False,
            reranker_backend="local",
            reranker_rollout_percentage=100,
            emergency_disabled=False,
            semantic_cache_enabled=False,
            semantic_cache_threshold=0.95,
            corpus_version=1,
            hyde_enabled=False,
            hyde_rollout_percentage=0,
            crag_enabled=True,
            crag_rollout_percentage=100,
            crag_web_enabled=False,
            crag_shadow_enabled=True,
        )
    )
    vector_repository = _FakeVectorRepository(chunks)
    strategy = DenseRetrievalStrategy(vector_repository=cast(VectorRepository, vector_repository))
    service = RAGService(
        embedding_client=cast(EmbeddingClient, _FakeEmbeddingClient()),
        retrieval_strategies={strategy.name: strategy},
        llm_client=cast(LLMClient, _FakeLLMClient()),
        cache=QueryCacheService(_InMemoryCacheBackend(), CacheSettings()),
        default_retrieval_mode=strategy.name,
        corrective_retriever=DynamicCorrectiveRetriever(
            delegate=cast(CorrectiveRetriever, delegate)
        ),
        crag_enabled=True,
        config_store=config_store,
        crag_telemetry=telemetry,  # type: ignore[arg-type]
    )

    service.answer("q", top_k=1)

    assert len(telemetry.calls) == 1
    assert telemetry.calls[0]["cohort"] == "shadow"
    assert telemetry.calls[0]["served"] is False


def test_fleet_telemetry_is_not_called_on_rollout_control_bypass() -> None:
    chunks = _chunks("a")
    delegate = _FakeCorrectiveDelegate()
    telemetry = _FakeCRAGTelemetry()
    config_store = RagRuntimeConfigStore(
        RagRuntimeConfig(
            reranking_enabled=False,
            reranker_backend="local",
            reranker_rollout_percentage=100,
            emergency_disabled=False,
            semantic_cache_enabled=False,
            semantic_cache_threshold=0.95,
            corpus_version=1,
            hyde_enabled=False,
            hyde_rollout_percentage=0,
            crag_enabled=True,
            crag_rollout_percentage=0,
            crag_web_enabled=False,
        )
    )
    vector_repository = _FakeVectorRepository(chunks)
    strategy = DenseRetrievalStrategy(vector_repository=cast(VectorRepository, vector_repository))
    service = RAGService(
        embedding_client=cast(EmbeddingClient, _FakeEmbeddingClient()),
        retrieval_strategies={strategy.name: strategy},
        llm_client=cast(LLMClient, _FakeLLMClient()),
        cache=QueryCacheService(_InMemoryCacheBackend(), CacheSettings()),
        default_retrieval_mode=strategy.name,
        corrective_retriever=DynamicCorrectiveRetriever(
            delegate=cast(CorrectiveRetriever, delegate)
        ),
        crag_enabled=True,
        config_store=config_store,
        crag_telemetry=telemetry,  # type: ignore[arg-type]
    )

    service.answer("q", top_k=1)

    assert telemetry.calls == []
    assert delegate.calls == []


def test_crag_metadata_disabled_when_crag_enabled_is_false() -> None:
    chunks = _chunks("a")
    delegate = _FakeCorrectiveDelegate()
    service, _ = _crag_service(chunks, delegate, crag_enabled=False)

    response = service.answer("q", top_k=1)

    assert response.metadata.crag.enabled is False
    assert response.metadata.crag.applied is False
    assert delegate.calls == []
    # Even disabled, evidence is identity-wrapped from retrieved chunks.
    assert [e.text for e in response.metadata.final_evidence] == ["a"]
