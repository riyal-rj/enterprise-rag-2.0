"""Self-reflection-specific integration tests for RAGService.answer().

Mirrors test_rag_service_crag.py's shape: uses the real
``DynamicSelfReflectionEngine`` (production adapter) wrapped around a raw
fake delegate, rather than a hand-rolled ``PlannedSelfReflectionEngine``
fake - the private fallback config store RAGService builds when no
``config_store`` is passed seeds ``self_reflective_rollout_percentage=100``,
so ``DynamicSelfReflectionEngine.plan`` deterministically resolves
"treatment" for every query whenever ``self_reflective_enabled=True``,
letting these tests ignore rollout sampling entirely (see
test_dynamic_self_reflection.py for the cohort-decision unit tests
themselves).
"""

from __future__ import annotations

from typing import cast

from qdrant_client.models import SparseVector

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
from app.rag_services.reflection.dynamic_self_reflection import DynamicSelfReflectionEngine
from app.rag_services.reflection.reflection import ReflectionAction, SelfReflectionOutcome
from app.rag_services.retrieval_strategy import DenseRetrievalStrategy, RetrievalStrategy
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
    def __init__(self, answer: str = "the initial answer") -> None:
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


class _FakeCorrectiveDelegate:
    def __init__(self, *, outcome_factory=None) -> None:  # type: ignore[no-untyped-def]
        self._outcome_factory = outcome_factory
        self.calls: list[dict[str, object]] = []

    @property
    def cache_namespace(self) -> str:
        return "crag:fake:v1"

    def correct(
        self, question: str, chunks: list[RetrievedChunk], *, allow_web: bool
    ) -> CRAGOutcome:
        self.calls.append({"question": question, "chunks": list(chunks), "allow_web": allow_web})
        if self._outcome_factory is not None:
            return self._outcome_factory(question, chunks)
        return CRAGOutcome(
            evidence=local_evidence(chunks), decision=CRAGDecision.CORRECT, applied=True
        )


class _FakeReflectionDelegate:
    """Delegate-level SelfReflectionEngine fake - _reflection_service wraps
    this in the real DynamicSelfReflectionEngine, same as production."""

    def __init__(self, *, outcome_factory=None, raise_error: bool = False) -> None:  # type: ignore[no-untyped-def]
        self._outcome_factory = outcome_factory
        self._raise_error = raise_error
        self.calls: list[dict[str, object]] = []

    @property
    def cache_namespace(self) -> str:
        return "self-reflection:fake:v1"

    def reflect(
        self,
        question: str,
        evidence: tuple[EvidenceChunk, ...],
        initial_answer: str,
        augmenter: object,
        *,
        allow_retrieval: bool = True,
    ) -> SelfReflectionOutcome:
        self.calls.append(
            {
                "question": question,
                "evidence": evidence,
                "initial_answer": initial_answer,
                "allow_retrieval": allow_retrieval,
            }
        )
        if self._raise_error:
            raise RuntimeError("boom")
        if self._outcome_factory is not None:
            return self._outcome_factory(question, evidence, initial_answer)
        return SelfReflectionOutcome(
            answer=initial_answer,
            evidence=evidence,
            applied=True,
            accepted=True,
            final_action=ReflectionAction.ACCEPT,
            iterations=1,
            additional_retrievals=0,
        )


def _reflection_service(
    results: list[RetrievedChunk],
    reflection_delegate: object,
    *,
    self_reflective_enabled: bool = True,
    llm_client: _FakeLLMClient | None = None,
    corrective_delegate: object | None = None,
    crag_enabled: bool = False,
    config_store: RagRuntimeConfigStore | None = None,
    self_reflection_telemetry: object | None = None,
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
        self_reflection_engine=DynamicSelfReflectionEngine(
            delegate=reflection_delegate  # type: ignore[arg-type]
        ),
        self_reflective_enabled=self_reflective_enabled,
    )
    if corrective_delegate is not None:
        kwargs["corrective_retriever"] = DynamicCorrectiveRetriever(
            delegate=cast(CorrectiveRetriever, corrective_delegate)
        )
        kwargs["crag_enabled"] = crag_enabled
    if config_store is not None:
        kwargs["config_store"] = config_store
    if self_reflection_telemetry is not None:
        kwargs["self_reflection_telemetry"] = self_reflection_telemetry
    service = RAGService(**kwargs)  # type: ignore[arg-type]
    return service, vector_repository


def _chunks(*texts: str) -> list[RetrievedChunk]:
    return [RetrievedChunk(text=t, source=f"{t}.pdf", score=0.9) for t in texts]


def test_reflection_sees_crag_evidence_and_initial_answer() -> None:
    chunks = _chunks("a")
    delegate = _FakeReflectionDelegate()
    llm_client = _FakeLLMClient(answer="draft answer")
    service, _ = _reflection_service(chunks, delegate, llm_client=llm_client)

    service.answer("q", top_k=1)

    assert delegate.calls[0]["question"] == "q"
    assert delegate.calls[0]["initial_answer"] == "draft answer"
    assert [e.text for e in delegate.calls[0]["evidence"]] == ["a"]


def test_reflection_never_runs_when_crag_already_abstained() -> None:
    chunks = _chunks("a")

    def crag_factory(question: str, chunks: list[RetrievedChunk]) -> CRAGOutcome:
        return CRAGOutcome(
            evidence=local_evidence(chunks),
            decision=CRAGDecision.INCORRECT,
            applied=True,
            abstain=True,
        )

    corrective_delegate = _FakeCorrectiveDelegate(outcome_factory=crag_factory)
    reflection_delegate = _FakeReflectionDelegate()
    service, _ = _reflection_service(
        chunks,
        reflection_delegate,
        corrective_delegate=corrective_delegate,
        crag_enabled=True,
    )

    response = service.answer("q", top_k=1)

    assert reflection_delegate.calls == []
    assert response.metadata.self_reflection.applied is False
    assert response.metadata.self_reflection.bypass_reason == "crag_abstained"


def test_final_answer_and_evidence_come_from_the_accepted_reflection_outcome() -> None:
    chunks = _chunks("raw chunk text")

    def reflection_factory(
        question: str, evidence: tuple[EvidenceChunk, ...], initial_answer: str
    ) -> SelfReflectionOutcome:
        return SelfReflectionOutcome(
            answer="revised, fully-cited answer",
            evidence=(
                EvidenceChunk(
                    text="revised evidence text",
                    source="b.pdf",
                    page_number=None,
                    retrieval_score=0.9,
                    origin=EvidenceOrigin.POLICY,
                ),
            ),
            applied=True,
            accepted=True,
            final_action=ReflectionAction.ACCEPT,
            iterations=2,
            additional_retrievals=1,
        )

    delegate = _FakeReflectionDelegate(outcome_factory=reflection_factory)
    service, _ = _reflection_service(chunks, delegate)

    response = service.answer("q", top_k=1)

    assert response.answer == "revised, fully-cited answer"
    assert [e.text for e in response.metadata.final_evidence] == ["revised evidence text"]
    assert response.sources == ["b.pdf"]
    assert response.metadata.self_reflection.iterations == 2
    assert response.metadata.self_reflection.additional_retrievals == 1


def test_accepted_reflection_response_is_cached() -> None:
    chunks = _chunks("a")
    delegate = _FakeReflectionDelegate()
    service, _ = _reflection_service(chunks, delegate)

    first = service.answer("q", top_k=1)
    calls_after_first = len(delegate.calls)
    second = service.answer("q", top_k=1)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(delegate.calls) == calls_after_first


def test_fallback_reflection_response_is_not_cached() -> None:
    chunks = _chunks("a")
    delegate = _FakeReflectionDelegate(raise_error=True)
    service, _ = _reflection_service(chunks, delegate)

    first = service.answer("q", top_k=1)
    second = service.answer("q", top_k=1)

    assert first.cache_hit is False
    assert second.cache_hit is False
    assert first.metadata.self_reflection.fallback is True
    assert len(delegate.calls) == 2


def test_abstain_reflection_response_is_not_cached() -> None:
    chunks = _chunks("a")

    def reflection_factory(
        question: str, evidence: tuple[EvidenceChunk, ...], initial_answer: str
    ) -> SelfReflectionOutcome:
        return SelfReflectionOutcome(
            answer="The supplied approved evidence is insufficient.",
            evidence=evidence,
            applied=True,
            accepted=False,
            final_action=ReflectionAction.ABSTAIN,
            iterations=2,
            additional_retrievals=1,
            abstain=True,
        )

    delegate = _FakeReflectionDelegate(outcome_factory=reflection_factory)
    service, _ = _reflection_service(chunks, delegate)

    first = service.answer("q", top_k=1)
    second = service.answer("q", top_k=1)

    assert first.cache_hit is False
    assert second.cache_hit is False
    assert len(delegate.calls) == 2


def test_reflection_metadata_disabled_when_self_reflective_enabled_is_false() -> None:
    chunks = _chunks("a")
    delegate = _FakeReflectionDelegate()
    service, _ = _reflection_service(chunks, delegate, self_reflective_enabled=False)

    response = service.answer("q", top_k=1)

    assert response.metadata.self_reflection.enabled is False
    assert response.metadata.self_reflection.applied is False
    assert delegate.calls == []
    # The unreflected answer is still returned - reflection being off must
    # not change baseline behavior.
    assert response.answer == "the initial answer"


def test_control_and_treatment_cohorts_use_isolated_caches() -> None:
    chunks = _chunks("a")
    delegate = _FakeReflectionDelegate()
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
            crag_enabled=False,
            crag_rollout_percentage=0,
            crag_web_enabled=False,
            self_reflective_enabled=True,
            self_reflective_rollout_percentage=50,
        )
    )
    service, _ = _reflection_service(chunks, delegate, config_store=config_store)

    control_question = next(
        q
        for q in (f"q{i}" for i in range(50))
        if service._self_reflection.plan(q, config_store.current, enabled=True).cohort == "control"
    )
    treatment_question = next(
        q
        for q in (f"q{i}" for i in range(50))
        if service._self_reflection.plan(q, config_store.current, enabled=True).cohort
        == "treatment"
    )

    control_key = service._cache_key(
        control_question,
        1,
        service._self_reflection.plan(
            control_question, config_store.current, enabled=True
        ).cache_namespace,
    )
    treatment_key = service._cache_key(
        treatment_question,
        1,
        service._self_reflection.plan(
            treatment_question, config_store.current, enabled=True
        ).cache_namespace,
    )

    assert control_key != treatment_key


def test_bounded_evidence_augmentation_uses_dense_retrieval_only() -> None:
    """The augmenter RAGService hands the engine must not reach the web or
    rerank - it's plain retrieval via the strategy already resolved for this
    request, over the same corpus - see _retrieve_reflection_evidence."""
    initial_chunks = _chunks("initial")
    delegate = _FakeReflectionDelegate()
    service, vector_repository = _reflection_service(initial_chunks, delegate)
    strategy = DenseRetrievalStrategy(vector_repository=cast(VectorRepository, vector_repository))

    evidence = service._retrieve_reflection_evidence("augmented query", strategy=strategy, top_k=5)

    assert [e.text for e in evidence] == ["initial"]  # same fake vector repo results
    assert all(e.origin is EvidenceOrigin.POLICY for e in evidence)
    assert vector_repository.search_calls[-1]["top_k"] == 5


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
    delegate = _FakeReflectionDelegate()
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
            crag_enabled=False,
            crag_rollout_percentage=0,
            crag_web_enabled=False,
            self_reflective_enabled=True,
            self_reflective_rollout_percentage=100,
        )
    )
    service, _ = _reflection_service(chunks, delegate, config_store=config_store)
    config_store.read_count = 0  # reset after construction-time reads, if any

    service.answer("q", top_k=1)

    assert config_store.read_count == 1


def test_fallback_with_flagged_claims_forces_deterministic_abstention() -> None:
    """A reflection failure alone must not be enough to serve an unverified
    draft - if the untouched pre-reflection answer trips the existing
    deterministic unsupported-claim check, the request must abstain
    instead, even though self-reflection itself never got to run."""
    chunks = _chunks("a")
    delegate = _FakeReflectionDelegate(raise_error=True)
    llm_client = _FakeLLMClient(
        answer="Customers must never receive any international wire transfers."
    )
    service, _ = _reflection_service(chunks, delegate, llm_client=llm_client)

    response = service.answer("q", top_k=1)

    assert response.metadata.self_reflection.fallback is True
    assert response.metadata.self_reflection.abstain is True
    assert response.answer != "Customers must never receive any international wire transfers."
    assert "internal error" in response.answer.lower()
    assert response.cache_hit is False


def test_fallback_abstains_even_without_flagged_claims() -> None:
    """Regression: a reflection failure must abstain unconditionally, not
    only when the lexical absolute-claim checker happens to flag something.
    That checker only catches sentences using absolute-claim language
    ("must", "cannot", ...) - a fabricated-but-plainly-worded claim (e.g.
    "the policy permits withdrawals up to INR 10,000") would otherwise pass
    it unflagged and get served with no verification at all, since
    self-reflection never actually ran to check it against evidence."""
    chunks = _chunks("a")
    delegate = _FakeReflectionDelegate(raise_error=True)
    llm_client = _FakeLLMClient(
        answer="The policy permits withdrawals up to INR 10,000 without further verification."
    )
    service, _ = _reflection_service(chunks, delegate, llm_client=llm_client)

    response = service.answer("q", top_k=1)

    assert response.metadata.self_reflection.fallback is True
    assert response.metadata.self_reflection.abstain is True
    assert "INR 10,000" not in response.answer
    assert "internal error" in response.answer.lower()
    assert response.cache_hit is False


def _shadow_config_store(*, retrieval_enabled: bool = False) -> RagRuntimeConfigStore:
    return RagRuntimeConfigStore(
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
            crag_enabled=False,
            crag_rollout_percentage=0,
            crag_web_enabled=False,
            self_reflective_enabled=True,
            self_reflective_rollout_percentage=0,
            self_reflective_shadow_enabled=True,
            self_reflective_retrieval_enabled=retrieval_enabled,
        )
    )


def test_shadow_mode_runs_reflection_but_never_serves_its_answer() -> None:
    chunks = _chunks("a")

    def reflection_factory(
        question: str, evidence: tuple[EvidenceChunk, ...], initial_answer: str
    ) -> SelfReflectionOutcome:
        return SelfReflectionOutcome(
            answer="shadow-only revised answer",
            evidence=(
                EvidenceChunk(
                    text="shadow-only evidence",
                    source="a.pdf",
                    page_number=None,
                    retrieval_score=0.9,
                    origin=EvidenceOrigin.POLICY,
                ),
            ),
            applied=True,
            accepted=True,
            final_action=ReflectionAction.ACCEPT,
            iterations=1,
            additional_retrievals=0,
        )

    delegate = _FakeReflectionDelegate(outcome_factory=reflection_factory)
    llm_client = _FakeLLMClient(answer="the baseline answer")
    config_store = _shadow_config_store()
    service, _ = _reflection_service(
        chunks, delegate, llm_client=llm_client, config_store=config_store
    )

    response = service.answer("q", top_k=1)

    # The delegate was actually invoked (observation) ...
    assert len(delegate.calls) == 1
    # ... but its answer/evidence never reached the response.
    assert response.answer == "the baseline answer"
    assert [e.text for e in response.metadata.final_evidence] == ["a"]
    assert response.metadata.self_reflection.bypass_reason == "shadow_not_served"
    assert response.metadata.self_reflection.applied is False


def test_shadow_mode_never_allows_retrieval_even_when_retrieval_enabled() -> None:
    chunks = _chunks("a")
    delegate = _FakeReflectionDelegate()
    config_store = _shadow_config_store(retrieval_enabled=True)
    service, _ = _reflection_service(chunks, delegate, config_store=config_store)

    service.answer("q", top_k=1)

    assert delegate.calls[0]["allow_retrieval"] is False


def test_shadow_response_is_cacheable_like_the_baseline_it_actually_serves() -> None:
    """A shadow-cohort response IS the baseline (unreflected) answer - shadow
    never changes what's served (see the previous test) - so it caches
    exactly like a baseline answer would, same as CRAG's shadow cohort. A
    cache hit on a repeated identical question skips re-observation for
    *that* question, which is fine: it already contributed one shadow data
    point, and shadow's job is aggregate observation, not exhaustive
    per-query coverage."""
    chunks = _chunks("a")
    delegate = _FakeReflectionDelegate()
    config_store = _shadow_config_store()
    service, _ = _reflection_service(chunks, delegate, config_store=config_store)

    first = service.answer("q", top_k=1)
    second = service.answer("q", top_k=1)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(delegate.calls) == 1


class _FakeSelfReflectionTelemetry:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def record_attempt(
        self,
        *,
        cohort: str,
        final_action: str | None,
        fallback: bool,
        abstain: bool,
        iterations: int,
        additional_retrievals: int,
        duration_ms: float,
        usage_tokens: int,
        served: bool,
    ) -> None:
        self.calls.append(
            {
                "cohort": cohort,
                "final_action": final_action,
                "fallback": fallback,
                "abstain": abstain,
                "served": served,
            }
        )


def test_fleet_telemetry_records_a_served_treatment_attempt() -> None:
    chunks = _chunks("a")
    delegate = _FakeReflectionDelegate()
    telemetry = _FakeSelfReflectionTelemetry()
    service, _ = _reflection_service(chunks, delegate, self_reflection_telemetry=telemetry)

    service.answer("q", top_k=1)

    assert len(telemetry.calls) == 1
    assert telemetry.calls[0]["cohort"] == "treatment"
    assert telemetry.calls[0]["served"] is True
    assert telemetry.calls[0]["final_action"] == "accept"


def test_fleet_telemetry_records_an_unserved_shadow_attempt() -> None:
    chunks = _chunks("a")
    delegate = _FakeReflectionDelegate()
    telemetry = _FakeSelfReflectionTelemetry()
    config_store = _shadow_config_store()
    service, _ = _reflection_service(
        chunks, delegate, config_store=config_store, self_reflection_telemetry=telemetry
    )

    service.answer("q", top_k=1)

    assert len(telemetry.calls) == 1
    assert telemetry.calls[0]["cohort"] == "shadow"
    assert telemetry.calls[0]["served"] is False


def test_fleet_telemetry_is_not_called_on_rollout_control_bypass() -> None:
    chunks = _chunks("a")
    delegate = _FakeReflectionDelegate()
    telemetry = _FakeSelfReflectionTelemetry()
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
            crag_enabled=False,
            crag_rollout_percentage=0,
            crag_web_enabled=False,
            self_reflective_enabled=True,
            self_reflective_rollout_percentage=0,
        )
    )
    service, _ = _reflection_service(
        chunks, delegate, config_store=config_store, self_reflection_telemetry=telemetry
    )

    service.answer("q", top_k=1)

    assert telemetry.calls == []
    assert delegate.calls == []
