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
via ``PipelineProfile.search_mode`` as its ``retrieval_mode`` override -
dense/sparse/hybrid all run for real. Reranking/HyDE/CRAG/self-reflective
profiles skip cleanly instead: none of those exist in the pipeline yet
(see ``app.rag_services.rag_service``'s module docstring), so silently
ignoring the flag would produce misleading pass/fail results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Protocol

from app.core.config import get_settings
from app.eval.profiles import PipelineProfile
from app.eval.schemas import Intent
from app.rag_services.rag_service import RAGService


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
            get_llm_client,
            get_retrieval_strategies,
        )
        from app.services.query_cache_service import NoOpCacheBackend, QueryCacheService

        return RAGService(
            embedding_client=get_embedding_client(),
            retrieval_strategies=get_retrieval_strategies(),
            llm_client=get_llm_client(),
            cache=QueryCacheService(NoOpCacheBackend(), get_settings().cache),
            default_retrieval_mode=get_default_retrieval_mode(),
            allowed_retrieval_modes=None,
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
        if flags.enable_hyde or flags.enable_rerank or flags.enable_crag or flags.enable_self_reflective:
            raise SkippedIntent(
                f"{flags.name}: HyDE/reranking/CRAG/self-reflective aren't implemented "
                "in the pipeline yet, only search_mode (dense/sparse/hybrid) is wired"
            )

        response = self._rag_service.answer(
            question, top_k=flags.top_k, retrieval_mode=flags.search_mode
        )
        chunks = [
            RetrievedChunk(text=c.text, source=c.source) for c in response.metadata.retrieved_chunks
        ]
        return InvokeResponse(answer=response.answer, sources=response.sources), chunks
