"""Basic RAG answer generation: embed, dense-search, generate, cache.

The full pipeline this is meant to grow into - self-reflective answer
refinement, SQL intent routing, prompt-injection spotlighting - layers on
top of this. HyDE query expansion, reranking, and CRAG evidence correction
are wired (see ``answer()``'s reranking/HyDE/CRAG stages): HyDE only ever
replaces the *dense retrieval vector*, never the sparse query, reranker
query, or answer-LLM question, and it fails open to the original-query
embedding on any LLM/schema/embedding/fusion error. CRAG grades the final
reranked/retrieved chunks (never a HyDE hypothesis, never a pre-rerank
pool), extractively refines them, and only escalates to allowlisted
public-regulatory web search for an explicitly in-scope query - see
``app.rag_services.crag`` for the full contract and non-negotiables. No
answer refinement loop, no prompt-injection defense yet.

Reranking, HyDE, and CRAG are each driven by a ``PlannedReranker``/
``PlannedQueryTransformer``/``PlannedCorrectiveRetriever`` (see
``app.rag_services.reranker.reranker`` / ``app.rag_services.hyde.query_transformer`` /
``app.rag_services.crag``): ``answer()`` captures exactly one
``RagRuntimeConfig`` snapshot per request (see
``app.rag_services.rag_runtime_config``) and calls each collaborator's
``plan()`` once, up front, to decide that request's cohort
(disabled/control/treatment) before any cache lookup - then threads the
same plan through cache-namespace construction, execution, and metrics.
This is what makes a concurrent admin config update unable to split one
request across two cohorts, and what keeps the semantic (paraphrase) cache
from ever serving a control-cohort query an answer generated under a
different query's treatment-cohort retrieval, or vice versa.

Caching follows the same cache-aside shape as
``app.core.llm.embedding_client``: check the cache first, compute and
write back on a miss, treat a cache read/write failure as a miss/no-op
rather than propagate it - the cache is an optimization, not a
correctness requirement.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Protocol

from app.core.exceptions import HybridRetrievalDisabledError
from app.core.llm.chat_client import LLMClient
from app.core.llm.embedding_client import EmbeddingClient
from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.claim_checker import find_unsupported_claims
from app.rag_services.confidence_scorer import compute_confidence_breakdown
from app.rag_services.crag import (
    CRAGOutcome,
    CRAGTelemetry,
    EvidenceChunk,
    EvidenceOrigin,
    NoOpCRAGTelemetry,
    PlannedCorrectiveRetriever,
    PlannedNoOpCorrectiveRetriever,
    local_evidence,
)
from app.rag_services.fusion.embedding_fusion import mean_pool_and_normalize
from app.rag_services.hyde.query_transformer import (
    NoOpQueryTransformer,
    PlannedNoOpQueryTransformer,
    PlannedQueryTransformer,
    QueryTransformOutcome,
)
from app.rag_services.rag_runtime_config import RagRuntimeConfig, RagRuntimeConfigStore
from app.rag_services.reflection.reflection import (
    EvidenceAugmenter,
    PlannedNoOpSelfReflectionEngine,
    PlannedSelfReflectionEngine,
    ReflectionAction,
    SelfReflectionOutcome,
)
from app.rag_services.reranker.reranker import PlannedNoOpReranker, PlannedReranker, ReRankedChunk
from app.rag_services.retrieval_strategy import RetrievalStrategy
from app.repositories.semantic_cache_repository import SemanticQueryCache
from app.schemas.chat import (
    ChatResponse,
    CRAGMetadata,
    EvidencePreview,
    HyDEMetadata,
    RerankingMetadata,
    ResponseMetadata,
    RetrievedChunkPreview,
    SelfReflectionMetadata,
)
from app.services.query_cache_service import CacheTier, QueryCacheService

logger = logging.getLogger(__name__)


class _RagServiceMetricsRecorder(Protocol):
    """Narrow collaborator - satisfied by ``app.services.rag_metrics_service.RagMetricsService``
    without this module needing to import it. Semantic-cache-lookup recording
    lives here, not in ``QdrantSemanticQueryCache.find_candidates``, because
    only this class knows the *confirmed* outcome: whether a candidate's
    exact-match Redis entry actually still exists, not just whether Qdrant
    found a similarity-clearing embedding match (see
    ``_check_semantic_cache``). HyDE-attempt recording lives here rather
    than in ``DynamicQueryTransformer`` because only ``RAGService`` observes
    the *complete* generation-embedding-fusion pipeline - see ``answer()``'s
    HyDE stage."""

    def record_semantic_cache_lookup(self, *, hit: bool) -> None: ...
    def record_hyde_attempt(
        self, *, duration_ms: float, fallback: bool, usage_tokens: int
    ) -> None: ...
    def record_hyde_bypass(self, *, reason: str) -> None: ...
    def record_crag_outcome(
        self,
        *,
        duration_ms: float,
        decision: str | None,
        fallback: bool,
        abstain: bool,
        web_used: bool,
        usage_tokens: int,
    ) -> None: ...
    def record_crag_bypass(self, *, reason: str) -> None: ...
    def record_crag_shadow_decision(
        self,
        *,
        decision: str | None,
        fallback: bool,
        abstain: bool,
        web_used: bool,
        evidence_count: int,
        duration_ms: float,
        usage_tokens: int,
    ) -> None: ...
    def record_self_reflection_outcome(
        self,
        *,
        duration_ms: float,
        final_action: str,
        fallback: bool,
        iterations: int,
        additional_retrievals: int,
        usage_tokens: int,
    ) -> None: ...
    def record_self_reflection_bypass(self, *, reason: str) -> None: ...


class _NoOpRagServiceMetrics:
    """Default when no metrics collaborator is supplied (e.g. the eval
    harness, which doesn't inject production metrics) - lets every call site
    record unconditionally instead of guarding on ``self._metrics is not None``."""

    def record_semantic_cache_lookup(self, *, hit: bool) -> None:
        pass

    def record_hyde_attempt(self, *, duration_ms: float, fallback: bool, usage_tokens: int) -> None:
        pass

    def record_hyde_bypass(self, *, reason: str) -> None:
        pass

    def record_crag_outcome(
        self,
        *,
        duration_ms: float,
        decision: str | None,
        fallback: bool,
        abstain: bool,
        web_used: bool,
        usage_tokens: int,
    ) -> None:
        pass

    def record_crag_bypass(self, *, reason: str) -> None:
        pass

    def record_crag_shadow_decision(
        self,
        *,
        decision: str | None,
        fallback: bool,
        abstain: bool,
        web_used: bool,
        evidence_count: int,
        duration_ms: float,
        usage_tokens: int,
    ) -> None:
        pass

    def record_self_reflection_outcome(
        self,
        *,
        duration_ms: float,
        final_action: str,
        fallback: bool,
        iterations: int,
        additional_retrievals: int,
        usage_tokens: int,
    ) -> None:
        pass

    def record_self_reflection_bypass(self, *, reason: str) -> None:
        pass


_SYSTEM_PROMPT = """
You are an enterprise banking-policy RAG assistant. Answer exclusively from the supplied policy context.

ANSWERING RULES

1. Give the direct operational answer first. Do not begin with a long explanation.

2. Answer every part of the question separately. If the question asks whether a customer can receive or send transactions, address receiving and sending independently.

3. Before generating the answer, internally identify:

   * Triggering facts
   * Mandatory actions
   * Conditional actions
   * Thresholds and aggregation rules
   * Deadlines and their starting events
   * Escalation and reporting requirements
   * Exceptions
   * Information not stated in the retrieved context

4. Preserve policy conditions exactly. Do not convert “if,” “where,” “unless,” “may,” or “subject to” into unconditional conclusions.

5. Distinguish between:

   * Mandatory now
   * Mandatory only if a stated condition is satisfied
   * Reasonable inference
   * Not established by the supplied context

6. Do not treat a reasonable inference as an explicit policy rule. If the context states “debit freeze,” conclude that account debits are restricted. Do not automatically conclude that incoming credits are either allowed or blocked unless the context explicitly states this.

7. A conditional policy requirement is not “context silence.” For example, if STR/SAR filing is required when suspicion is confirmed, state the conditional filing requirement and its deadlines.

8. For multi-policy scenarios, provide every applicable control needed to answer the question. Check for:

   * KYC or EDD requirements
   * Transaction review requirements
   * CTR aggregation and thresholds
   * STR/SAR escalation and filing deadlines
   * Information-security escalation
   * Privacy or regulatory notification

9. Do not include unrelated facts merely because they are true. A high-risk customer’s annual Re-KYC cycle should not be added unless it changes the required action in the scenario.

10. Use the response format that best fits the query:

* Binary decision: direct answer followed by evidence boundaries
* Multi-action scenario: ordered action list
* Deadline question: chronological table
* Comparison question: comparison table
* Insufficient evidence: explicit abstention identifying the missing policy or procedure

11. Cite every material action, threshold, deadline, exception, and conclusion using:
    [Policy ID, version, page, section]

12. Cite only evidence actually used in the answer. Do not cite unrelated or merely similar retrieved documents.

13. If the required answer is not established by the retrieved context, state:
    “The supplied policy context does not explicitly determine this point.”

14. Never invent a policy, deadline, threshold, permission, prohibition, exception, or operational process.

15. Before returning the answer, verify:

* Every part of the question has been answered.
* Every mandatory claim has supporting evidence.
* Every conditional rule remains conditional.
* All relevant deadlines and thresholds are included.
* No unrelated policy information has been added.

CONFIDENCE RULE

Do not generate an arbitrary numeric confidence score. Confidence should be calculated externally from retrieval quality, claim-evidence coverage, citation entailment, completeness, and calibration data. If a confidence score is supplied by the application, explain any evidence limitation that materially affects it.

"""

_SELF_REFLECTION_FALLBACK_ABSTENTION = (
    "An internal error prevented full verification of this answer against "
    "approved evidence, and part of the draft could not be confirmed. "
    "Please retry your question."
)


class _CallableEvidenceAugmenter:
    """Adapts a bound closure to the ``EvidenceAugmenter`` protocol -
    ``RAGService.answer`` builds one per request from a closure over that
    request's already-captured ``strategy``/``config`` (see
    ``RAGService._retrieve_reflection_evidence``), so the reflection engine
    never needs a reference to ``RAGService`` itself and can't recursively
    call back into ``answer()``."""

    def __init__(self, callback: Callable[[str], tuple[EvidenceChunk, ...]]) -> None:
        self._callback = callback

    def retrieve(self, query: str) -> tuple[EvidenceChunk, ...]:
        return self._callback(query)


class RAGService:
    """Retrieve-then-generate RAG answering, with response caching."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        retrieval_strategies: Mapping[str, RetrievalStrategy],
        llm_client: LLMClient,
        cache: QueryCacheService,
        default_retrieval_mode: str,
        allowed_retrieval_modes: frozenset[str] | None = None,
        reranker: PlannedReranker | None = None,
        reranker_initial_top_k: int = 20,
        reranking_enabled: bool = False,
        semantic_cache: SemanticQueryCache | None = None,
        semantic_cache_enabled: bool = False,
        metrics: _RagServiceMetricsRecorder | None = None,
        query_transformer: PlannedQueryTransformer | None = None,
        hyde_enabled: bool = False,
        corrective_retriever: PlannedCorrectiveRetriever | None = None,
        crag_enabled: bool = False,
        config_store: RagRuntimeConfigStore | None = None,
        crag_telemetry: CRAGTelemetry | None = None,
        self_reflection_engine: PlannedSelfReflectionEngine | None = None,
        self_reflective_enabled: bool = False,
        reflection_retrieval_top_k: int = 5,
    ) -> None:
        if default_retrieval_mode not in retrieval_strategies:
            raise ValueError(
                f"default_retrieval_mode {default_retrieval_mode!r} is not among "
                f"the available retrieval_strategies: {sorted(retrieval_strategies)}"
            )
        # None means "everything registered is allowed" - the eval harness's
        # unrestricted RAGService instance uses this, since it must be able
        # to run hybrid/sparse for real before the rollout flag is ever
        # flipped on for real HTTP traffic (see app.eval.invokers).
        if allowed_retrieval_modes is None:
            allowed_retrieval_modes = frozenset(retrieval_strategies)
        if default_retrieval_mode not in allowed_retrieval_modes:
            raise ValueError(
                f"default_retrieval_mode {default_retrieval_mode!r} is not among "
                f"allowed_retrieval_modes: {sorted(allowed_retrieval_modes)}"
            )
        self._embedding_client = embedding_client
        self._retrieval_strategies = retrieval_strategies
        self._llm_client = llm_client
        self._cache = cache
        self._default_retrieval_mode = default_retrieval_mode
        self._allowed_retrieval_modes = allowed_retrieval_modes
        self._reranker = reranker or PlannedNoOpReranker()
        self._reranker_initial_top_k = reranker_initial_top_k
        self._semantic_cache = semantic_cache
        self._metrics = metrics or _NoOpRagServiceMetrics()
        self._query_transformer = query_transformer or PlannedNoOpQueryTransformer()
        self._corrective_retriever = corrective_retriever or PlannedNoOpCorrectiveRetriever()
        self._self_reflection = self_reflection_engine or PlannedNoOpSelfReflectionEngine()
        self._reflection_retrieval_top_k = reflection_retrieval_top_k
        # Fleet-wide (cross-worker/cross-process) exporter - see
        # app.rag_services.crag.telemetry. Distinct from `metrics` above
        # (process-local, powers the admin panel): this is the seam for a
        # real Prometheus/OpenTelemetry backend, and is a no-op unless a
        # deployment actually wires one in.
        self._crag_telemetry = crag_telemetry or NoOpCRAGTelemetry()
        # A caller that cares about atomic cross-object config application
        # (app.api.deps.get_rag_service) passes the same config_store this
        # process's DynamicReranker/DynamicQueryTransformer read from (see
        # app.rag_services.rag_runtime_config). Callers that don't (mostly
        # tests, and the eval harness's standalone service instance) get a
        # private store seeded from the reranking_enabled/
        # semantic_cache_enabled/hyde_enabled kwargs, preserving the old
        # constructor-arg ergonomics for a service instance nothing else
        # needs to observe.
        self._config_store = config_store or RagRuntimeConfigStore(
            RagRuntimeConfig(
                reranking_enabled=reranking_enabled,
                reranker_backend="local",
                reranker_rollout_percentage=100,
                emergency_disabled=False,
                semantic_cache_enabled=semantic_cache_enabled,
                semantic_cache_threshold=0.95,
                corpus_version=1,
                hyde_enabled=hyde_enabled,
                hyde_rollout_percentage=100,
                crag_enabled=crag_enabled,
                crag_rollout_percentage=100,
                crag_web_enabled=False,
                crag_shadow_enabled=False,
                self_reflective_enabled=self_reflective_enabled,
                self_reflective_rollout_percentage=100,
            )
        )

    def answer(
        self,
        question: str,
        top_k: int = 5,
        retrieval_mode: str | None = None,
        *,
        reranking_enabled: bool | None = None,
        hyde_enabled: bool | None = None,
        crag_enabled: bool | None = None,
        self_reflective_enabled: bool | None = None,
    ) -> ChatResponse:
        """Answer ``question``.

        ``reranking_enabled``/``hyde_enabled``/``crag_enabled``/
        ``self_reflective_enabled`` override the
        instance-level defaults (from the config snapshot captured below,
        seeded from the centralized RAG ops config in production) for this
        call only - ``None`` (the default) defers to the instance. This is
        what lets the eval harness exercise the pipeline with any
        combination of flags through the same service instance/config (see
        ``app.eval.invokers.ServiceInvoker``) instead of needing a
        separately-wired ``RAGService`` per combination.
        """
        # Captured once, up front - the ONLY read of the shared store in
        # this whole call. Every cohort decision (reranking, HyDE), the
        # cache namespace, and the semantic-cache threshold below all
        # derive from this one snapshot, so a config update landing
        # mid-request can't split this request across two generations of
        # config. See app.rag_services.rag_runtime_config.
        config = self._config_store.current
        effective_reranking_enabled = (
            config.reranking_enabled if reranking_enabled is None else reranking_enabled
        )
        effective_hyde_enabled = config.hyde_enabled if hyde_enabled is None else hyde_enabled
        effective_crag_enabled = config.crag_enabled if crag_enabled is None else crag_enabled
        effective_self_reflective_enabled = (
            config.self_reflective_enabled
            if self_reflective_enabled is None
            else self_reflective_enabled
        )

        mode = retrieval_mode or self._default_retrieval_mode
        if mode in self._retrieval_strategies:
            if mode not in self._allowed_retrieval_modes:
                raise HybridRetrievalDisabledError(mode)
            strategy = self._retrieval_strategies[mode]
        else:
            strategy = self._retrieval_strategies[self._default_retrieval_mode]

        candidate_top_k = top_k
        # corpus_version is folded in here - not just an audit-trail number
        # - so a replaced/re-seeded corpus makes every previously-cached
        # exact-match answer and semantic-cache pointer (both partitioned by
        # this same cache_namespace) unreachable the moment the version
        # bumps, even if the explicit Redis/Qdrant clear() calls that
        # accompany a corpus mutation (see
        # app.services.corpus_cache_invalidation_service) fail or are
        # skipped.
        cache_namespace = f"{strategy.cache_namespace}:corpus={config.corpus_version}"

        # Reranking/HyDE cohorts are decided once, here, from the single
        # `config` snapshot above and the query text - before any cache
        # lookup - so a control-cohort query and a treatment-cohort query
        # under the same rollout percentage can never share a cache
        # namespace (a plain "rollout=30%" segment would collide the two;
        # rerank_plan.cache_namespace/hyde_plan.cache_namespace instead
        # encode which cohort *this* query actually landed in). Both plans
        # are always computed (cheap, no I/O), but their segment is only
        # folded into cache_namespace when the feature is actually enabled -
        # when it's off entirely there's no cohort ambiguity to disambiguate
        # (every request behaves identically), so the namespace stays
        # exactly what it was before either feature existed.
        rerank_plan = self._reranker.plan(question, config, enabled=effective_reranking_enabled)
        if effective_reranking_enabled:
            cache_namespace = f"{cache_namespace}:{rerank_plan.cache_namespace}"
            if rerank_plan.cohort == "treatment":
                # Only the treatment cohort ever widens the pool - a
                # control-cohort query (rollout-bypassed) must retrieve
                # exactly `top_k`, same as if reranking were off entirely,
                # or it would pay retrieval cost for a wider pool a reranker
                # that never runs has no use for. candidates=N is part of
                # the key, not just top_k/reranker name: widening the pool
                # (e.g. 20 -> 50) can change which chunk the reranker picks
                # first even though top_k and the reranker itself are
                # unchanged, so a stale narrower-pool answer must not be
                # served under the new configuration.
                candidate_top_k = max(top_k, self._reranker_initial_top_k)
                cache_namespace = f"{cache_namespace}:candidates={candidate_top_k}"

        hyde_can_run = effective_hyde_enabled and strategy.requires_dense_embedding
        hyde_plan = self._query_transformer.plan(question, config, enabled=hyde_can_run)
        if hyde_can_run:
            cache_namespace = f"{cache_namespace}:{hyde_plan.cache_namespace}"

        # CRAG cohort is decided here too, from the same single `config`
        # snapshot, before any cache lookup - same reasoning as
        # rerank_plan/hyde_plan above: a control-cohort query and a
        # treatment-cohort query under the same rollout percentage must not
        # share a cache namespace, since treatment can rewrite the evidence
        # (and even add web sources) that reaches the answer LLM.
        crag_plan = self._corrective_retriever.plan(
            question, config, enabled=effective_crag_enabled
        )
        if effective_crag_enabled:
            cache_namespace = f"{cache_namespace}:{crag_plan.cache_namespace}"

        # Self-reflection cohort is decided here too, from the same single
        # `config` snapshot, before any cache lookup - same reasoning as
        # rerank_plan/hyde_plan/crag_plan above: a control-cohort query and a
        # treatment-cohort query under the same rollout percentage must not
        # share a cache namespace, since treatment can revise the answer (and
        # even widen the evidence) that's ultimately served.
        self_reflection_plan = self._self_reflection.plan(
            question, config, enabled=effective_self_reflective_enabled
        )
        if effective_self_reflective_enabled:
            # reflection_retrieval_top_k is a RAGService-level knob (used
            # only when building the bounded evidence augmenter below, see
            # _retrieve_reflection_evidence) that none of the reflection
            # collaborators' own cache_namespace strings know about - fold it
            # in here, the same way candidate_top_k is folded in for
            # reranking above, so a deploy that changes it can't serve a
            # stale answer computed under a different retrieval width.
            cache_namespace = (
                f"{cache_namespace}:{self_reflection_plan.cache_namespace}:"
                f"reflection_top_k={self._reflection_retrieval_top_k}"
            )

        cache_key = self._cache_key(question, top_k, cache_namespace)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached.model_copy(update={"cache_hit": True})

        # The original embedding is the semantic-cache identity - computing
        # it before HyDE lets a semantic hit avoid all HyDE LLM and
        # hypothesis-embedding cost. `similarity_threshold` comes from the
        # same captured `config`, not a second, independent store read
        # inside the semantic-cache repository - see
        # app.repositories.semantic_cache_repository.
        original_query_embedding: list[float] | None = None
        if (
            strategy.requires_dense_embedding
            and config.semantic_cache_enabled
            and self._semantic_cache is not None
        ):
            original_query_embedding = self._embedding_client.embed_texts([question])[0]
            semantic_hit = self._check_semantic_cache(
                original_query_embedding, cache_namespace, top_k, config.semantic_cache_threshold
            )
            if semantic_hit is not None:
                return semantic_hit.model_copy(update={"cache_hit": True})

        hyde_outcome = NoOpQueryTransformer(reason=hyde_plan.bypass_reason or "disabled").transform(
            question
        )
        query_embedding: list[float] | None = None

        if strategy.requires_dense_embedding:
            if hyde_plan.cohort == "treatment":
                hyde_started = time.perf_counter()
                hyde_outcome = self._query_transformer.execute(question, hyde_plan)
                if hyde_outcome.applied:
                    try:
                        hypothesis_embeddings = self._embedding_client.embed_texts(
                            list(hyde_outcome.retrieval_texts)
                        )
                        query_embedding = mean_pool_and_normalize(hypothesis_embeddings)
                    except Exception as exc:  # noqa: BLE001 - retrieval must remain available
                        logger.warning(
                            "rag.hyde_embedding_fallback",
                            extra={"error_type": type(exc).__name__},
                        )
                        hyde_outcome = QueryTransformOutcome(
                            retrieval_texts=(question,),
                            backend=hyde_outcome.backend,
                            applied=False,
                            fallback=True,
                            bypass_reason="embedding_or_fusion_error",
                            usage_tokens=hyde_outcome.usage_tokens,
                            duration_ms=hyde_outcome.duration_ms,
                        )
                # Total stage latency - generation (execute) + hypothesis
                # embedding + fusion - recorded once, after the complete
                # pipeline: the delegate itself never sets duration_ms, and
                # recording the metric right after generation (before
                # embedding/fusion could still fail) would misreport an
                # embedding/fusion failure as a successful HyDE attempt.
                hyde_outcome = replace(
                    hyde_outcome, duration_ms=(time.perf_counter() - hyde_started) * 1000
                )
                self._metrics.record_hyde_attempt(
                    duration_ms=hyde_outcome.duration_ms,
                    fallback=hyde_outcome.fallback,
                    usage_tokens=hyde_outcome.usage_tokens,
                )
            elif hyde_plan.bypass_reason in ("rollout", "emergency_disabled"):
                # "disabled" (feature toggle fully off) isn't a canary
                # bucket worth counting - only rollout-control and
                # emergency bypasses are, matching HyDEMetricsSnapshot's
                # two named counters (see app.services.rag_metrics_service).
                self._metrics.record_hyde_bypass(reason=hyde_plan.bypass_reason)

            if query_embedding is None:
                if original_query_embedding is None:
                    original_query_embedding = self._embedding_client.embed_texts([question])[0]
                query_embedding = original_query_embedding

        chunks = strategy.retrieve(
            query_text=question,  # always original - hybrid's sparse branch uses this
            query_embedding=query_embedding,
            top_k=candidate_top_k,
        )
        # Exactly what `chunks` would be without reranking - same slice,
        # same order, same .score values - so confidence scoring's
        # retrieval-strength component stays meaningful (and comparable to
        # the non-reranking case) even after chunks below is reordered by
        # a rerank score that has nothing to do with retrieval strength.
        retrieval_ordered_top_k = chunks[:top_k]

        reranked = False
        fallback_occurred = False
        rerank_candidate_count = 0
        rerank_backend = rerank_plan.reported_backend
        rerank_usage_tokens: int | None = None
        rerank_items: tuple[ReRankedChunk, ...] | None = None
        if rerank_plan.cohort == "treatment" and chunks:
            rerank_candidate_count = len(chunks)
            rerank_started = time.perf_counter()
            outcome = self._reranker.execute(
                rerank_plan, query=question, candidates=chunks, top_k=top_k
            )
            duration_ms = (time.perf_counter() - rerank_started) * 1000
            chunks = [item.chunk for item in outcome.items]
            rerank_items = outcome.items
            reranked = outcome.applied
            fallback_occurred = outcome.fallback
            rerank_backend = outcome.backend
            rerank_usage_tokens = outcome.usage_tokens

            log_fields = {
                "backend": outcome.backend,
                "applied": outcome.applied,
                "fallback": outcome.fallback,
                "candidate_count": rerank_candidate_count,
                "result_count": len(chunks),
                "duration_ms": round(duration_ms, 2),
            }
            if outcome.fallback:
                logger.warning("rag.reranking_fallback", extra=log_fields)
            else:
                logger.debug("rag.reranking_applied", extra=log_fields)

        # CRAG grades/refines/corrects the exact evidence that would
        # otherwise reach the answer LLM - the final reranked/retrieved
        # `chunks`, never a pre-rerank pool or a HyDE hypothesis (see module
        # docstring). ``execute`` always runs, even when CRAG is disabled or
        # this query landed in the rollout control cohort: in every one of
        # those cases ``crag_plan.cohort != "treatment"`` routes to
        # ``PlannedNoOpCorrectiveRetriever``, which wraps `chunks` as-is via
        # ``local_evidence`` - so evidence-context construction below never
        # needs to branch on whether CRAG actually ran.
        crag_started = time.perf_counter()
        if crag_plan.cohort == "shadow":
            # Observe-only: the real corrective retriever actually runs (for
            # measurement - see DynamicCorrectiveRetriever.execute treating
            # "shadow" like "treatment"), but its evidence is never served.
            # A shadow miss/regression must never change what a real user
            # sees, and web must stay impossible in shadow regardless of
            # crag_web_enabled - crag_plan.allow_web is always False here
            # (see DynamicCorrectiveRetriever.plan).
            shadow_outcome = self._corrective_retriever.execute(question, chunks, crag_plan)
            shadow_outcome = replace(
                shadow_outcome, duration_ms=(time.perf_counter() - crag_started) * 1000
            )
            self._metrics.record_crag_shadow_decision(
                decision=(shadow_outcome.decision.value if shadow_outcome.decision else None),
                fallback=shadow_outcome.fallback,
                abstain=shadow_outcome.abstain,
                web_used=shadow_outcome.web_used,
                evidence_count=len(shadow_outcome.evidence),
                duration_ms=shadow_outcome.duration_ms,
                usage_tokens=shadow_outcome.usage_tokens,
            )
            self._crag_telemetry.record_attempt(
                cohort=crag_plan.cohort,
                decision=(shadow_outcome.decision.value if shadow_outcome.decision else None),
                fallback=shadow_outcome.fallback,
                abstain=shadow_outcome.abstain,
                web_used=shadow_outcome.web_used,
                duration_ms=shadow_outcome.duration_ms,
                usage_tokens=shadow_outcome.usage_tokens,
                served=False,
            )
            logger.debug(
                "rag.crag_shadow_observed",
                extra={
                    "decision": (
                        shadow_outcome.decision.value if shadow_outcome.decision else None
                    ),
                    "applied": shadow_outcome.applied,
                    "fallback": shadow_outcome.fallback,
                    "abstain": shadow_outcome.abstain,
                    "web_used": shadow_outcome.web_used,
                    "evidence_count": len(shadow_outcome.evidence),
                    "duration_ms": round(shadow_outcome.duration_ms, 2),
                },
            )
            crag_outcome = CRAGOutcome(
                evidence=local_evidence(chunks),
                decision=None,
                applied=False,
                bypass_reason="shadow_not_served",
            )
        else:
            crag_outcome = self._corrective_retriever.execute(question, chunks, crag_plan)
            if crag_plan.cohort == "treatment":
                crag_outcome = replace(
                    crag_outcome, duration_ms=(time.perf_counter() - crag_started) * 1000
                )
                self._metrics.record_crag_outcome(
                    duration_ms=crag_outcome.duration_ms,
                    decision=(crag_outcome.decision.value if crag_outcome.decision else None),
                    fallback=crag_outcome.fallback,
                    abstain=crag_outcome.abstain,
                    web_used=crag_outcome.web_used,
                    usage_tokens=crag_outcome.usage_tokens,
                )
                self._crag_telemetry.record_attempt(
                    cohort=crag_plan.cohort,
                    decision=(crag_outcome.decision.value if crag_outcome.decision else None),
                    fallback=crag_outcome.fallback,
                    abstain=crag_outcome.abstain,
                    web_used=crag_outcome.web_used,
                    duration_ms=crag_outcome.duration_ms,
                    usage_tokens=crag_outcome.usage_tokens,
                    served=True,
                )
                log_fields = {
                    "decision": (crag_outcome.decision.value if crag_outcome.decision else None),
                    "applied": crag_outcome.applied,
                    "fallback": crag_outcome.fallback,
                    "abstain": crag_outcome.abstain,
                    "web_used": crag_outcome.web_used,
                    "evidence_count": len(crag_outcome.evidence),
                    "duration_ms": round(crag_outcome.duration_ms, 2),
                }
                if crag_outcome.fallback:
                    logger.warning("rag.crag_fallback_applied", extra=log_fields)
                else:
                    logger.debug("rag.crag_applied", extra=log_fields)
            elif crag_plan.bypass_reason in ("rollout", "emergency_disabled"):
                # "disabled" (feature toggle fully off) isn't a canary
                # bucket worth counting - only rollout-control and
                # emergency bypasses are, matching HyDE's identical
                # distinction above.
                self._metrics.record_crag_bypass(reason=crag_plan.bypass_reason)

        # Abstention is a deterministic response, never an LLM guess filling
        # an evidence gap - see module docstring's non-negotiable #4 and
        # ``ProductionCorrectiveRetriever.correct``'s internal-policy branch.
        if crag_outcome.abstain:
            if crag_outcome.web_used:
                answer_text = (
                    "The available approved internal and regulatory evidence "
                    "is insufficient to determine this point."
                )
            else:
                answer_text = (
                    "The supplied approved internal evidence is insufficient "
                    "to determine this point. No external evidence was used."
                )
        else:
            user_message = (
                f"{self._build_evidence_context(crag_outcome.evidence)}\n\nQuestion: {question}"
            )
            answer_text = self._llm_client.generate(_SYSTEM_PROMPT, user_message).text

        # Self-reflection critiques/revises the initial answer against
        # CRAG-approved evidence, and may perform one bounded additional
        # retrieval (never a web search, never a re-entrant call into this
        # method) - see app.rag_services.reflection. It never sees a HyDE
        # hypothesis, and never runs when CRAG already abstained - there is
        # no generated answer to critique, and the abstention string is
        # already the safe terminal response.
        reflection_started = time.perf_counter()
        if (
            effective_self_reflective_enabled
            and self_reflection_plan.cohort == "treatment"
            and not crag_outcome.abstain
        ):
            augmenter: EvidenceAugmenter = _CallableEvidenceAugmenter(
                lambda query: self._retrieve_reflection_evidence(
                    query, strategy=strategy, top_k=self._reflection_retrieval_top_k
                )
            )
            self_reflection_outcome = self._self_reflection.execute(
                question, crag_outcome.evidence, answer_text, augmenter, self_reflection_plan
            )
            self_reflection_outcome = replace(
                self_reflection_outcome,
                duration_ms=(time.perf_counter() - reflection_started) * 1000,
            )
            self._metrics.record_self_reflection_outcome(
                duration_ms=self_reflection_outcome.duration_ms,
                final_action=self_reflection_outcome.final_action.value,
                fallback=self_reflection_outcome.fallback,
                iterations=self_reflection_outcome.iterations,
                additional_retrievals=self_reflection_outcome.additional_retrievals,
                usage_tokens=self_reflection_outcome.usage_tokens,
            )
            log_fields = {
                "final_action": self_reflection_outcome.final_action.value,
                "accepted": self_reflection_outcome.accepted,
                "fallback": self_reflection_outcome.fallback,
                "abstain": self_reflection_outcome.abstain,
                "iterations": self_reflection_outcome.iterations,
                "additional_retrievals": self_reflection_outcome.additional_retrievals,
                "duration_ms": round(self_reflection_outcome.duration_ms, 2),
            }
            if self_reflection_outcome.fallback:
                logger.warning("rag.self_reflection_fallback_applied", extra=log_fields)
            else:
                logger.debug("rag.self_reflection_applied", extra=log_fields)
        else:
            bypass_reason = self_reflection_plan.bypass_reason
            if crag_outcome.abstain and effective_self_reflective_enabled:
                bypass_reason = "crag_abstained"
            self_reflection_outcome = SelfReflectionOutcome(
                answer=answer_text,
                evidence=crag_outcome.evidence,
                applied=False,
                accepted=True,
                final_action=ReflectionAction.ACCEPT,
                iterations=0,
                additional_retrievals=0,
                bypass_reason=bypass_reason,
            )
            if effective_self_reflective_enabled and self_reflection_plan.bypass_reason in (
                "rollout",
                "emergency_disabled",
            ):
                # "disabled" (feature toggle fully off) isn't a canary bucket
                # worth counting - only rollout-control and emergency
                # bypasses are, matching HyDE/CRAG's identical distinction.
                self._metrics.record_self_reflection_bypass(
                    reason=self_reflection_plan.bypass_reason
                )

        # A reflection *failure* (LLM/schema/retrieval error) falls back to
        # the untouched pre-reflection draft - exactly the answer that would
        # have been served had self-reflection never run. But an operator
        # who turned self-reflection on for a banking-policy system opted
        # into an extra layer of assurance against unsupported claims;
        # silently reverting to "no safety net" on a technical hiccup
        # defeats that. Gate the fallback draft on the same deterministic,
        # zero-LLM-cost check the response already computes for every answer
        # (see flagged_claims below) - if it flags anything, force a
        # deterministic abstention instead of serving unverified content.
        # fallback=True already keeps this out of the cache either way (see
        # pipeline_fallback below); this only changes what's served to the
        # caller for *this* request.
        if self_reflection_outcome.fallback:
            fallback_claims = find_unsupported_claims(
                self_reflection_outcome.answer,
                self._evidence_as_retrieved_chunks(self_reflection_outcome.evidence),
            )
            if fallback_claims:
                logger.warning(
                    "rag.self_reflection_fallback_unsafe_draft",
                    extra={"flagged_claim_count": len(fallback_claims)},
                )
                self_reflection_outcome = replace(
                    self_reflection_outcome,
                    answer=_SELF_REFLECTION_FALLBACK_ABSTENTION,
                    abstain=True,
                )

        # Everything downstream (confidence, claim-checking, sources,
        # final-evidence preview) must see exactly what was actually
        # returned - the post-reflection answer/evidence when reflection ran
        # and accepted/revised, or the pre-reflection CRAG answer/evidence
        # otherwise. Never the pre-reflection answer paired with
        # post-reflection evidence (or vice versa).
        answer_text = self_reflection_outcome.answer
        final_evidence = self_reflection_outcome.evidence

        # Confidence and claim-checking must see exactly what was returned -
        # CRAG's (and, if it ran, self-reflection's) final evidence, not the
        # pre-CRAG retrieved/reranked `chunks` - or a corrected/revised
        # answer would be scored against evidence it was never actually
        # generated from.
        final_evidence_chunks = self._evidence_as_retrieved_chunks(final_evidence)

        confidence = compute_confidence_breakdown(
            final_evidence_chunks,
            answer_text,
            retrieval_mode=strategy.name,
            # Preserve original retrieval-strength calculation separately.
            retrieval_ordered_chunks=retrieval_ordered_top_k,
        )
        logger.debug(
            "rag.confidence_computed",
            extra={
                "confidence": confidence.total,
                "evidence_coverage": confidence.evidence_coverage,
                "faithfulness": confidence.faithfulness,
                "retrieval_strength": confidence.retrieval_strength,
                "citation_precision": confidence.citation_precision,
                "answerability": confidence.answerability,
            },
        )

        flagged_claims = find_unsupported_claims(answer_text, final_evidence_chunks)
        if flagged_claims:
            # Never log raw questions or unsupported claim text - both can
            # contain sensitive policy/customer detail.
            logger.warning(
                "rag.unsupported_claim_flagged",
                extra={
                    "flagged_claim_count": len(flagged_claims),
                    "crag_decision": (
                        crag_outcome.decision.value if crag_outcome.decision is not None else None
                    ),
                },
            )

        response = ChatResponse(
            answer=answer_text,
            sources=sorted({self._public_source_identifier(item) for item in final_evidence}),
            confidence=confidence.total,
            metadata=ResponseMetadata(
                route="rag",
                retrieval_mode=strategy.name,
                hyde=HyDEMetadata(
                    enabled=hyde_can_run,
                    applied=hyde_outcome.applied,
                    fallback=hyde_outcome.fallback,
                    backend=hyde_outcome.backend,
                    hypothesis_count=(
                        len(hyde_outcome.retrieval_texts) if hyde_outcome.applied else 0
                    ),
                    usage_tokens=hyde_outcome.usage_tokens,
                    duration_ms=round(hyde_outcome.duration_ms, 2),
                    bypass_reason=hyde_outcome.bypass_reason,
                ),
                reranking=RerankingMetadata(
                    enabled=effective_reranking_enabled,
                    applied=reranked,
                    fallback=fallback_occurred,
                    backend=rerank_backend,
                    candidate_count=rerank_candidate_count,
                    usage_tokens=rerank_usage_tokens,
                ),
                crag=CRAGMetadata(
                    enabled=effective_crag_enabled,
                    applied=crag_outcome.applied,
                    decision=(crag_outcome.decision.value if crag_outcome.decision else None),
                    fallback=crag_outcome.fallback,
                    abstain=crag_outcome.abstain,
                    web_used=crag_outcome.web_used,
                    evidence_count=len(crag_outcome.evidence),
                    usage_tokens=crag_outcome.usage_tokens,
                    duration_ms=round(crag_outcome.duration_ms, 2),
                    bypass_reason=crag_outcome.bypass_reason,
                ),
                self_reflection=SelfReflectionMetadata(
                    enabled=effective_self_reflective_enabled,
                    applied=self_reflection_outcome.applied,
                    accepted=self_reflection_outcome.accepted,
                    final_action=self_reflection_outcome.final_action.value,
                    iterations=self_reflection_outcome.iterations,
                    additional_retrievals=self_reflection_outcome.additional_retrievals,
                    fallback=self_reflection_outcome.fallback,
                    abstain=self_reflection_outcome.abstain,
                    support_level=(
                        self_reflection_outcome.support_level.value
                        if self_reflection_outcome.support_level
                        else None
                    ),
                    answer_relevance=self_reflection_outcome.answer_relevance,
                    citation_completeness=self_reflection_outcome.citation_completeness,
                    utility=self_reflection_outcome.utility,
                    usage_tokens=self_reflection_outcome.usage_tokens,
                    duration_ms=round(self_reflection_outcome.duration_ms, 2),
                    bypass_reason=self_reflection_outcome.bypass_reason,
                ),
                retrieved_chunks=self._build_chunk_previews(chunks, rerank_items),
                final_evidence=self._build_evidence_previews(final_evidence),
                flagged_claims=flagged_claims,
            ),
        )

        # A fallback response is retrieval-order output produced *under
        # the reranked/HyDE/CRAG config's cache key* - caching it would let
        # later requests keep receiving the degraded answer even after the
        # reranker/HyDE/CRAG recovers, since nothing would ever invalidate
        # it (see module docstring: reads/writes are cache-aside, there's no
        # TTL tied to pipeline health). Simplest correct fix: never cache a
        # fallback, abstained, or web-corrected response at all, exact-match
        # or semantic - web state isn't versioned with corpus_version (see
        # CRAG blueprint's non-negotiable #4/#5), and an abstention is a
        # statement about *this* evidence set, not a durable answer.
        pipeline_fallback = (
            fallback_occurred
            or hyde_outcome.fallback
            or crag_outcome.fallback
            or crag_outcome.web_used
            or crag_outcome.abstain
            or self_reflection_outcome.fallback
            or self_reflection_outcome.abstain
            # Explicit, not merely implied by fallback/abstain above: today
            # every non-accepted outcome this package produces also sets one
            # of those two flags, but this must hold even if a future engine
            # implementation ever sets accepted=False without them - caching
            # an answer self-reflection didn't actually accept is never safe.
            or not self_reflection_outcome.accepted
        )
        if not pipeline_fallback:
            cache_written = self._set_cached(cache_key, response)
            if (
                cache_written
                and config.semantic_cache_enabled
                and self._semantic_cache is not None
                and original_query_embedding is not None
            ):
                self._record_semantic_cache(
                    original_query_embedding, cache_namespace, top_k, cache_key
                )
        return response

    def _retrieve_reflection_evidence(
        self,
        query: str,
        *,
        strategy: RetrievalStrategy,
        top_k: int,
    ) -> tuple[EvidenceChunk, ...]:
        """The one bounded additional retrieval self-reflection is allowed to
        perform (see ``app.rag_services.reflection``'s module docstring).
        Plain dense/sparse/hybrid retrieval via the same strategy already
        resolved for this request - no reranking, no web, no CRAG re-grade,
        and critically no call back into ``answer()`` - so this can't
        recurse and can't widen scope beyond the corpus this request was
        already allowed to search. ``query`` is expected to already have
        passed ``validate_reflection_query`` (the reflection engine does
        this before calling ``EvidenceAugmenter.retrieve``); any failure
        here (embedding/retrieval error) propagates to the engine's caller,
        where ``FailSafeSelfReflectionEngine`` treats it like any other
        reflection-stage failure."""
        query_embedding: list[float] | None = None
        if strategy.requires_dense_embedding:
            query_embedding = self._embedding_client.embed_texts([query])[0]
        chunks = strategy.retrieve(query_text=query, query_embedding=query_embedding, top_k=top_k)
        return local_evidence(chunks)

    def _build_chunk_previews(
        self,
        chunks: list[RetrievedChunk],
        rerank_items: tuple[ReRankedChunk, ...] | None,
    ) -> list[RetrievedChunkPreview]:
        if rerank_items is not None:
            return [
                RetrievedChunkPreview(
                    text=item.chunk.text,
                    source=item.chunk.source,
                    score=item.chunk.score,
                    page_number=item.chunk.page_number,
                    rerank_score=item.rerank_score,
                    original_rank=item.original_rank,
                )
                for item in rerank_items
            ]
        return [
            RetrievedChunkPreview(
                text=c.text, source=c.source, score=c.score, page_number=c.page_number
            )
            for c in chunks
        ]

    @staticmethod
    def _build_evidence_previews(
        evidence: tuple[EvidenceChunk, ...],
    ) -> list[EvidencePreview]:
        return [
            EvidencePreview(
                text=item.text,
                source=item.source,
                page_number=item.page_number,
                score=item.retrieval_score,
                origin=item.origin.value,
                canonical_url=item.canonical_url,
                retrieved_at_iso=item.retrieved_at_iso,
            )
            for item in evidence
        ]

    @staticmethod
    def _evidence_as_retrieved_chunks(
        evidence: tuple[EvidenceChunk, ...],
    ) -> list[RetrievedChunk]:
        """CRAG's final evidence, reshaped for the confidence/claim-checking
        code paths that only know about ``RetrievedChunk`` - preferring a
        web item's canonical URL as its ``source`` so those downstream
        checks (and the eval harness, see ``app.eval.invokers``) see the
        same trusted identifier the API response does."""
        return [
            RetrievedChunk(
                text=item.text,
                source=item.canonical_url or item.source,
                score=item.retrieval_score,
                page_number=item.page_number,
            )
            for item in evidence
        ]

    @staticmethod
    def _public_source_identifier(item: EvidenceChunk) -> str:
        if item.origin is EvidenceOrigin.REGULATORY_WEB and item.canonical_url is not None:
            return item.canonical_url
        return item.source

    def _build_evidence_context(self, evidence: tuple[EvidenceChunk, ...]) -> str:
        """Render CRAG-approved evidence for the answer LLM.

        ``evidence`` is always CRAG's output - identity-wrapped original
        chunks when CRAG is disabled/control (see ``PlannedNoOpCorrectiveRetriever``),
        graded/refined/possibly-web-augmented evidence when it's active.
        Metadata is JSON-serialized (not string-interpolated) so a source
        name or URL containing special characters can't corrupt the
        surrounding ``<evidence>`` delimiters; both ``QUESTION`` and this
        evidence text remain untrusted data per this module's system
        prompt, never instructions.
        """
        if not evidence:
            return "No relevant context was found."

        blocks: list[str] = []
        for index, item in enumerate(evidence, start=1):
            metadata = {
                "evidence_id": index,
                "source": item.source,
                "page_number": item.page_number,
                "origin": item.origin.value,
                "canonical_url": item.canonical_url,
                "retrieved_at": item.retrieved_at_iso,
            }
            blocks.append(f"<evidence metadata={json.dumps(metadata, ensure_ascii=True)}>")
            blocks.append(item.text)
            blocks.append("</evidence>")
        return "\n".join(blocks)

    def _cache_key(self, question: str, top_k: int, cache_namespace: str) -> str:
        normalized_question = " ".join(question.split())

        # v8: reflection/critic/reviser cache_namespace strings now include
        # every output-affecting setting (max_completion_tokens, max_attempts,
        # stage/total timeout, schema version), and reflection_retrieval_top_k
        # is folded into cache_namespace above - none of that was namespaced
        # under v7, so a v7-or-earlier key could silently collide across a
        # config change to any of those. On top of v7's self-reflection
        # cohort isolation, v6's CRAG cohort isolation, and v5's reranking/
        # HyDE cohort isolation - bumped so a v7-or-earlier key can't collide
        # with or mask a v8 key for the same question/config.
        raw_key = f"rag:v8:{cache_namespace}:{top_k}:{normalized_question}"

        return hashlib.sha256(raw_key.encode()).hexdigest()

    def _get_cached(self, key: str) -> ChatResponse | None:
        try:
            raw = self._cache.get(CacheTier.RAG_ANSWER, key)
        except Exception as exc:  # noqa: BLE001 - cache is an optimization, not a correctness requirement
            logger.warning("rag.cache_read_failed", extra={"error": str(exc)})
            return None
        return ChatResponse.model_validate_json(raw) if raw is not None else None

    def _set_cached(self, key: str, response: ChatResponse) -> bool:
        """Write ``response`` to the exact-match cache. Returns whether the
        write actually succeeded - callers that only want to index a
        semantic-cache pointer after a *confirmed* write (see ``answer()``)
        need this instead of assuming ``set`` succeeded."""
        try:
            self._cache.set(CacheTier.RAG_ANSWER, key, response.model_dump_json())
            return True
        except Exception as exc:  # noqa: BLE001 - cache is an optimization, not a correctness requirement
            logger.warning("rag.cache_write_failed", extra={"error": str(exc)})
            return False

    def _check_semantic_cache(
        self,
        query_embedding: list[float],
        cache_namespace: str,
        top_k: int,
        similarity_threshold: float,
    ) -> ChatResponse | None:
        """Try each semantic-cache candidate, nearest-first, until one's
        exact-match Redis entry is confirmed still live. Records the
        semantic-cache hit-rate metric exactly once per call, using that
        *confirmed* outcome - not whether Qdrant merely returned a
        similarity-clearing candidate, which can still be a stale pointer
        (see ``QdrantSemanticQueryCache.find_candidates``)."""
        hit: ChatResponse | None = None
        for candidate_key in self._find_semantic_cache_candidates(
            query_embedding, cache_namespace, top_k, similarity_threshold
        ):
            cached = self._get_cached(candidate_key)
            if cached is not None:
                hit = cached
                break
            # This candidate's answer expired/was cleared from Redis
            # since it was indexed - try the next-nearest match rather
            # than immediately falling through to a fresh generation.
        self._metrics.record_semantic_cache_lookup(hit=hit is not None)
        return hit

    def _find_semantic_cache_candidates(
        self,
        query_embedding: list[float],
        cache_namespace: str,
        top_k: int,
        similarity_threshold: float,
    ) -> list[str]:
        assert self._semantic_cache is not None  # guarded by caller
        try:
            return self._semantic_cache.find_candidates(
                query_embedding=query_embedding,
                cache_namespace=cache_namespace,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
        except Exception as exc:  # noqa: BLE001 - cache is an optimization, not a correctness requirement
            logger.warning("rag.semantic_cache_read_failed", extra={"error": str(exc)})
            return []

    def _record_semantic_cache(
        self, query_embedding: list[float], cache_namespace: str, top_k: int, cache_key: str
    ) -> None:
        assert self._semantic_cache is not None  # guarded by caller
        try:
            self._semantic_cache.record(
                query_embedding=query_embedding,
                cache_namespace=cache_namespace,
                top_k=top_k,
                cache_key=cache_key,
            )
        except Exception as exc:  # noqa: BLE001 - cache is an optimization, not a correctness requirement
            logger.warning("rag.semantic_cache_write_failed", extra={"error": str(exc)})
