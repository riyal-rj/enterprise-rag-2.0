from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from app.models.retrieved_chunk import RetrievedChunk

if TYPE_CHECKING:
    from app.rag_services.rag_runtime_config import RagRuntimeConfig, RerankerBackendName

logger = logging.getLogger(__name__)


@dataclass
class ReRankedChunk:
    chunk: RetrievedChunk
    original_rank: int
    rerank_score: float | None


@dataclass
class ReRankOutcome:
    items: tuple[ReRankedChunk, ...]
    backend: str
    applied: bool
    fallback: bool = False
    usage_tokens: int | None = None


@runtime_checkable
class ReRanker(Protocol):
    @property
    def name(self) -> str:
        """Human Readable backend name for metrics and response metadata"""
        ...

    @property
    def cache_namespace(self) -> str:
        """Stable identifier containing backend, model and implementation version."""
        ...

    def rerank(
        self, *, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> ReRankOutcome: ...


class NoOpReranker:
    @property
    def name(self) -> str:
        return "none"

    @property
    def cache_namespace(self) -> str:
        return "reranker:none:v1"

    def rerank(
        self, *, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> ReRankOutcome:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        items = tuple(
            ReRankedChunk(
                chunk=chunk,
                original_rank=rank,
                rerank_score=None,
            )
            for rank, chunk in enumerate(candidates[:top_k], start=1)
        )

        return ReRankOutcome(
            items=items,
            backend=self.name,
            applied=False,
        )


class FailOpenReranker:
    """Resilience decorator"""

    def __init__(self, delegate: ReRanker, fallback: ReRanker | None = None) -> None:
        self._delegate = delegate
        self._fallback = fallback

    @property
    def name(self) -> str:
        return self._delegate.name

    @property
    def cache_namespace(self) -> str:
        return f"fail-open:v1:{self._delegate.cache_namespace}"

    def rerank(
        self, *, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> ReRankOutcome:

        try:
            return self._delegate.rerank(query=query, candidates=candidates, top_k=top_k)
        except Exception:
            logger.exception(
                "Rerank failed, using retrieval ordering",
                extra={
                    "reranker": self._delegate.name,
                    "candidate_count": len(candidates),
                    "top_k": top_k,
                },
            )

            if self._fallback is None:
                items = tuple(
                    ReRankedChunk(
                        chunk=chunk,
                        original_rank=rank,
                        rerank_score=None,
                    )
                    for rank, chunk in enumerate(candidates[:top_k], start=1)
                )

                return ReRankOutcome(
                    items=items,
                    backend=self._delegate.name,
                    applied=False,
                    fallback=True,
                )

            fallback = self._fallback.rerank(
                query=query,
                candidates=candidates,
                top_k=top_k,
            )

            return ReRankOutcome(
                items=fallback.items,
                backend=self._delegate.name,
                applied=False,
                fallback=True,
            )


RerankCohort = Literal["disabled", "control", "treatment"]


@dataclass(frozen=True)
class RerankPlan:
    """One query's reranking cohort, decided once from a caller-supplied
    config snapshot and reused for cache-namespace construction and
    execution - never re-derived mid-request. ``backend_key`` is ``None``
    unless ``cohort == "treatment"``; ``reported_backend`` is always a
    human-readable name suitable for response metadata even when nothing
    ran."""

    cohort: RerankCohort
    backend_key: RerankerBackendName | None
    reported_backend: str
    bypass_reason: str | None
    cache_namespace: str


class PlannedReranker(Protocol):
    """Cohort-aware contract both the production (rollout/emergency-gated)
    and eval (always-on) reranking paths implement, so ``RAGService.answer``
    can call ``plan()``/``execute()`` uniformly regardless of which one it
    was given - see ``DynamicReranker`` (production, in
    ``app.rag_services.dynamic_reranker``) and ``StaticPlannedReranker``
    (eval, below)."""

    def plan(self, query: str, config: RagRuntimeConfig, *, enabled: bool) -> RerankPlan: ...

    def execute(
        self, plan: RerankPlan, *, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> ReRankOutcome: ...


class PlannedNoOpReranker:
    """``RAGService``'s default when no reranker is supplied at all (e.g. a
    bare unit-test instance) - always reports cohort=disabled."""

    def plan(self, query: str, config: RagRuntimeConfig, *, enabled: bool) -> RerankPlan:
        return RerankPlan("disabled", None, "none", "disabled", "rerank:none:reason=disabled")

    def execute(
        self, plan: RerankPlan, *, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> ReRankOutcome:
        return NoOpReranker().rerank(query=query, candidates=candidates, top_k=top_k)


class StaticPlannedReranker:
    """Eval-only adapter around ``get_reranker()``'s static
    ``FailOpenReranker``: always reranks for real when ``enabled`` is true,
    ignoring admin rollout%/emergency-disable state entirely - the same
    guarantee ``get_reranker``'s docstring already promises, now expressed
    through the same plan()/execute() shape production uses so
    ``RAGService`` doesn't need a branch for eval."""

    def __init__(self, delegate: ReRanker) -> None:
        self._delegate = delegate

    def plan(self, query: str, config: RagRuntimeConfig, *, enabled: bool) -> RerankPlan:
        if not enabled:
            return RerankPlan("disabled", None, "none", "disabled", "rerank:none:reason=disabled")
        return RerankPlan(
            "treatment",
            None,
            self._delegate.name,
            None,
            f"static:v1:{self._delegate.cache_namespace}",
        )

    def execute(
        self, plan: RerankPlan, *, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> ReRankOutcome:
        if plan.cohort != "treatment":
            return NoOpReranker().rerank(query=query, candidates=candidates, top_k=top_k)
        return self._delegate.rerank(query=query, candidates=candidates, top_k=top_k)
