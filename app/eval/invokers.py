"""Invokes the system under test (Strategy: transport into the pipeline).

``ServiceInvoker`` calls the RAG/SQL pipeline in-process (direct Python
call — what CI should use). An HTTP-based invoker for ``--mode api``
doesn't exist yet, matching ``run_ragas.main``'s own "Phase B" placeholder
for that mode.

Kept as a ``Protocol`` (not an ABC): every other swappable interface in
this codebase — ``RateLimiter``, ``PasswordHasher``, ``TokenIssuer``,
``CacheBackend``, ``UserRepository``, ``HealthCheck`` — is a ``Protocol``,
so implementations don't need to inherit from anything, just match the
shape. Switching this one to an ABC would be the odd one out.

``ServiceInvoker`` only attempts intents the pipeline can run headlessly.
SQL cases are excluded: Text2SQL requires a human-in-the-loop
``interrupt()`` approval step before execution (see the ``sql``-tagged
golden cases' notes in ``data/goldens.yaml``), which a batch eval run has
no one to answer. ``web_fallback`` is excluded too - a Tavily search
adapter exists (``app.services.web_search.search_web``), but nothing
wires its results into an answer-generation pipeline yet, so there's
nothing web-fallback-specific to actually invoke; running it would just
silently re-run ``RAGService.answer()`` against the policy corpus.
Unsupported intents are checked *before* calling the pipeline, so a case
is cleanly skipped with a clear reason rather than erroring partway
through a real call.

``_call_pipeline`` calls the real :class:`~app.rag_services.rag_service.RAGService`,
via ``PipelineProfile.search_mode`` as its ``retrieval_mode`` override,
``PipelineProfile.enable_rerank`` as a per-call ``reranking_enabled``
override, ``PipelineProfile.enable_hyde`` as a per-call ``hyde_enabled``
override, and ``PipelineProfile.enable_crag`` as a per-call
``crag_enabled`` override (see ``RAGService.answer``) - dense/sparse/hybrid,
reranking, HyDE, and CRAG all run for real, through
``app.api.deps.get_eval_hyde_transformer``/``get_eval_crag`` (never the
production admin-rollout-controlled transformer/corrective retriever - see
``_rag_service`` below). Self-reflective profiles still skip cleanly: it
doesn't exist in the pipeline yet (see ``app.rag_services.rag_service``'s
module docstring), so silently ignoring that flag would produce misleading
pass/fail results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Protocol

from app.core.config import get_settings
from app.core.exceptions import EvaluationPipelineError
from app.eval.profiles import PipelineProfile
from app.eval.schemas import Intent
from app.rag_services.query_transformer import StaticPlannedQueryTransformer
from app.rag_services.rag_service import RAGService
from app.rag_services.reranker import StaticPlannedReranker


class SkippedIntent(Exception):
    """Raised when a case can't be run against the current wiring.

    Not a failure — the caller buckets these separately from scored rows
    (see ``run_ragas.main``'s ``skipped`` list).
    """


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved context passage, in the shape RAGAS needs."""

    text: str
    source: str


@dataclass(frozen=True)
class InvokeResponse:
    """The system under test's final answer for one question."""

    answer: str
    sources: list[str] = field(default_factory=list)


class Invoker(Protocol):
    """Contract for reaching the system under test."""

    def invoke(
        self, question: str, flags: PipelineProfile, intent: Intent
    ) -> tuple[InvokeResponse, list[RetrievedChunk]]: ...


class ServiceInvoker:
    """In-process invoker, calling the real :class:`RAGService`.

    ``rag_service`` is normally left unset - the real one is built lazily
    (see ``_rag_service``, only constructed on first actual pipeline call,
    not at ``__init__`` time) so guard-check-only tests don't need to spin
    up the full DI chain (Qdrant, OpenAI, FastEmbed). Pass an explicit
    ``rag_service`` (e.g. a fake) to bypass that entirely, which is also
    how tests inject a fake without monkeypatching module-level DI functions.

    Deliberately unrestricted: constructed with ``allowed_retrieval_modes=None``
    (see ``RAGService.__init__``), bypassing the ``hybrid_search_enabled``
    rollout gate that a real HTTP ``/chat`` request is subject to (via
    ``app.api.deps.get_rag_service``) - eval must be able to run every mode
    for real ahead of that flag ever being flipped on.

    The wrapped ``RAGService`` is given a :class:`~app.services.query_cache_service.NoOpCacheBackend`,
    not the shared production cache - a case must actually execute the
    current retriever/config every run, not silently return a cached
    answer computed under a previous ``rrf_k``/candidate-count/corpus
    version (the cache key doesn't cover every tunable, so a stale hit
    would go unnoticed).
    """

    SUPPORTED_INTENTS = frozenset({Intent.RAG})

    def __init__(self, rag_service: RAGService | None = None) -> None:
        self._rag_service_override = rag_service

    @cached_property
    def _rag_service(self) -> RAGService:
        if self._rag_service_override is not None:
            return self._rag_service_override

        # Imported lazily (not at module level) so importing this module
        # doesn't require the full app DI chain to be importable/configured.
        from app.api.deps import (
            get_default_retrieval_mode,
            get_embedding_client,
            get_eval_crag,
            get_eval_hyde_transformer,
            get_llm_client,
            get_reranker,
            get_retrieval_strategies,
        )
        from app.services.query_cache_service import NoOpCacheBackend, QueryCacheService

        rag_settings = get_settings().rag
        return RAGService(
            embedding_client=get_embedding_client(),
            retrieval_strategies=get_retrieval_strategies(),
            llm_client=get_llm_client(),
            cache=QueryCacheService(NoOpCacheBackend(), get_settings().cache),
            default_retrieval_mode=get_default_retrieval_mode(),
            allowed_retrieval_modes=None,
            # Built with the same production reranker config regardless of
            # RAGFeatureSettings.reranking_enabled_by_default - flags.enable_rerank
            # (a per-call override, see _call_pipeline) is what actually
            # switches it on/off per case, so hybrid+rerank profiles get a
            # real reranker to compare against, not a silent NoOpReranker
            # that would make "ran with reranking on" a no-op in practice.
            # Wrapped in StaticPlannedReranker so RAGService's plan()/
            # execute() calls work uniformly whether it holds this eval
            # reranker or production's DynamicReranker - always reranks for
            # real when flags.enable_rerank is on, ignoring admin
            # rollout%/emergency-disable state (see StaticPlannedReranker's
            # docstring in app.rag_services.reranker).
            reranker=StaticPlannedReranker(get_reranker()),
            reranker_initial_top_k=rag_settings.reranker_initial_top_k,
            # Same reasoning as the reranker above, for HyDE: the eval-only
            # transformer (never the admin-rollout-controlled production
            # one), wrapped in StaticPlannedQueryTransformer so it always
            # attempts HyDE for real when flags.enable_hyde (a per-call
            # override, see _call_pipeline) is on - never inheriting an
            # admin's live rollout percentage or emergency-disable state.
            query_transformer=StaticPlannedQueryTransformer(get_eval_hyde_transformer()),
            hyde_enabled=False,
            # Same reasoning as the reranker/HyDE overrides above, for CRAG:
            # the eval-only corrective retriever (never the admin-rollout-
            # controlled production one), already wrapped in
            # StaticPlannedCorrectiveRetriever by get_eval_crag() so it
            # always attempts CRAG for real when flags.enable_crag (a
            # per-call override, see _call_pipeline) is on - never
            # inheriting an admin's live rollout percentage or
            # emergency-disable state. Web correction stays off for every
            # eval case (StaticPlannedCorrectiveRetriever.plan always sets
            # allow_web=False) so an offline run never depends on a live
            # Tavily call.
            corrective_retriever=get_eval_crag(),
            crag_enabled=False,
        )

    def invoke(
        self, question: str, flags: PipelineProfile, intent: Intent
    ) -> tuple[InvokeResponse, list[RetrievedChunk]]:
        if intent not in self.SUPPORTED_INTENTS:
            raise SkippedIntent(
                f"intent={intent.value} not supported in service mode "
                "(sql/hybrid need human-in-the-loop approval, not runnable headlessly; "
                "web_fallback has no implemented answer-generation pipeline yet)"
            )

        return self._call_pipeline(question, flags)

    def _call_pipeline(
        self, question: str, flags: PipelineProfile
    ) -> tuple[InvokeResponse, list[RetrievedChunk]]:
        if flags.enable_self_reflective:
            raise SkippedIntent(
                f"{flags.name}: self-reflective isn't implemented "
                "in the pipeline yet, only search_mode, reranking, HyDE, and CRAG are wired"
            )

        response = self._rag_service.answer(
            question,
            top_k=flags.top_k,
            retrieval_mode=flags.search_mode,
            reranking_enabled=flags.enable_rerank,
            hyde_enabled=flags.enable_hyde,
            crag_enabled=flags.enable_crag,
        )

        if flags.enable_hyde:
            hyde = response.metadata.hyde
            if not hyde.applied or hyde.fallback or hyde.bypass_reason is not None:
                # A case that asked for HyDE and silently got baseline
                # (non-HyDE) output would score as if HyDE were under test
                # when it never actually ran - fail the case instead of
                # quietly measuring the wrong pipeline. StaticPlannedQueryTransformer
                # ignores rollout/emergency state, so bypass_reason here can
                # only mean a real generation/embedding/fusion failure.
                raise EvaluationPipelineError(
                    f"{flags.name}: HyDE was requested but not applied "
                    f"(applied={hyde.applied}, fallback={hyde.fallback}, "
                    f"bypass_reason={hyde.bypass_reason!r})"
                )

        if flags.enable_crag:
            crag = response.metadata.crag
            if not crag.applied or crag.fallback:
                # Same reasoning as the HyDE check above: a case that asked
                # for CRAG and silently got baseline (ungraded) evidence
                # would score as if CRAG were under test when it never
                # actually ran cleanly.
                raise EvaluationPipelineError(
                    f"{flags.name}: CRAG was requested but not cleanly applied "
                    f"(applied={crag.applied}, fallback={crag.fallback}, "
                    f"bypass_reason={crag.bypass_reason!r})"
                )

        if flags.enable_crag and not response.metadata.final_evidence:
            # A CRAG-enabled case that produced no final evidence would
            # otherwise silently score RAGAS against an empty context list -
            # indistinguishable from "CRAG correctly found nothing" versus
            # "final_evidence metadata was never populated". Fail loudly
            # instead.
            raise EvaluationPipelineError(f"{flags.name}: CRAG returned no final-evidence metadata")

        # RAGAS must score exactly the context the answer model saw - CRAG's
        # final evidence when CRAG ran, never the pre-CRAG retrieved/
        # reranked chunks (see app.rag_services.rag_service.RAGService.answer).
        # A web item's canonical URL is preferred as `source` so its
        # ranked_sources entry matches ChatResponse.sources (see
        # RAGService._public_source_identifier).
        chunks = [
            RetrievedChunk(text=item.text, source=item.canonical_url or item.source)
            for item in response.metadata.final_evidence
        ]
        return InvokeResponse(answer=response.answer, sources=response.sources), chunks
