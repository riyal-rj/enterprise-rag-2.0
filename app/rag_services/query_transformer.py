"""Query-transformation contracts and safe decorators.

Two layers live here. The delegate layer (``QueryTransformer``,
``QueryTransformOutcome``, ``NoOpQueryTransformer``,
``FailOpenQueryTransformer``) is a single unconditional "transform this
query" call with no notion of admin-mutable rollout/emergency state - it's
what the raw LLM-backed ``HydeQueryTransformer`` and its failure-handling
wrapper implement.

The planned layer (``PlannedQueryTransformer`` and friends) sits on top:
``plan()`` decides a query's cohort from a single, caller-supplied
``RagRuntimeConfig`` snapshot (no internal state read), and ``execute()``
performs the transformation according to a previously-computed plan. This
is what lets ``RAGService.answer`` capture exactly one config snapshot per
request and thread the same cohort decision through cache-namespace
construction, execution, and metrics - see that module for why a second,
independent config read partway through a request is a real bug, not a
style preference. Both the production path (``DynamicQueryTransformer``)
and the eval harness (``StaticPlannedQueryTransformer`` below) implement
this same contract so ``RAGService`` never needs to know which one it has.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from app.rag_services.rag_runtime_config import RagRuntimeConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryTransformOutcome:
    """Result of preparing texts for dense retrieval.

    ``retrieval_texts`` are never evidence and must never be sent to the
    answer LLM. When ``applied`` is false, the tuple contains the original
    question so the caller can safely embed it.
    """

    retrieval_texts: tuple[str, ...]
    backend: str
    applied: bool
    fallback: bool = False
    bypass_reason: str | None = None
    usage_tokens: int = 0
    duration_ms: float = 0.0


class QueryTransformer(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def cache_namespace(self) -> str: ...

    def transform(self, query: str) -> QueryTransformOutcome: ...


class NoOpQueryTransformer:
    def __init__(self, *, reason: str = "disabled") -> None:
        self._reason = reason

    @property
    def name(self) -> str:
        return "none"

    @property
    def cache_namespace(self) -> str:
        return f"query-transform:none:reason={self._reason}"

    def transform(self, query: str) -> QueryTransformOutcome:
        return QueryTransformOutcome(
            retrieval_texts=(query,),
            backend=self.name,
            applied=False,
            bypass_reason=self._reason,
        )


class FailOpenQueryTransformer:
    """Convert any transformer failure into original-query retrieval."""

    def __init__(self, delegate: QueryTransformer) -> None:
        self._delegate = delegate

    @property
    def name(self) -> str:
        return self._delegate.name

    @property
    def cache_namespace(self) -> str:
        return self._delegate.cache_namespace

    def transform(self, query: str) -> QueryTransformOutcome:
        try:
            return self._delegate.transform(query)
        except Exception as exc:  # noqa: BLE001 - deliberate availability boundary
            logger.warning(
                "rag.hyde_fallback",
                extra={"backend": self._delegate.name, "error_type": type(exc).__name__},
            )
            return QueryTransformOutcome(
                retrieval_texts=(query,),
                backend=self._delegate.name,
                applied=False,
                fallback=True,
                bypass_reason="transform_error",
            )


HyDECohort = Literal["disabled", "control", "treatment"]


@dataclass(frozen=True)
class HyDEPlan:
    """One query's HyDE cohort, decided once from a caller-supplied config
    snapshot and reused for cache-namespace construction, execution, and
    metrics - never re-derived mid-request. ``bypass_reason`` is ``None``
    iff ``cohort == "treatment"``; otherwise one of "disabled",
    "emergency_disabled", or "rollout"."""

    cohort: HyDECohort
    bypass_reason: str | None
    cache_namespace: str


class PlannedQueryTransformer(Protocol):
    """Cohort-aware contract both the production (rollout/emergency-gated)
    and eval (always-on) HyDE paths implement, so ``RAGService.answer`` can
    call ``plan()``/``execute()`` uniformly regardless of which one it was
    given - see ``DynamicQueryTransformer`` (production, in
    ``app.rag_services.dynamic_query_transformer``) and
    ``StaticPlannedQueryTransformer`` (eval, below)."""

    def plan(self, query: str, config: RagRuntimeConfig, *, enabled: bool) -> HyDEPlan: ...

    def execute(self, query: str, plan: HyDEPlan) -> QueryTransformOutcome: ...


class PlannedNoOpQueryTransformer:
    """``RAGService``'s default when no query transformer is supplied at
    all (e.g. a bare unit-test instance) - always reports cohort=disabled
    and never calls an LLM."""

    def plan(self, query: str, config: RagRuntimeConfig, *, enabled: bool) -> HyDEPlan:
        return HyDEPlan("disabled", "disabled", "query-transform:none:reason=disabled")

    def execute(self, query: str, plan: HyDEPlan) -> QueryTransformOutcome:
        return NoOpQueryTransformer(reason=plan.bypass_reason or "disabled").transform(query)


class StaticPlannedQueryTransformer:
    """Eval-only adapter around ``get_eval_hyde_transformer()``'s existing
    ``FailOpenQueryTransformer``: always attempts HyDE for real when
    ``enabled`` is true, ignoring admin rollout%/emergency-disable state
    entirely - the same guarantee ``get_eval_hyde_transformer``'s docstring
    already promises, now expressed through the same plan()/execute() shape
    production uses so ``RAGService`` doesn't need a branch for eval."""

    def __init__(self, delegate: QueryTransformer) -> None:
        self._delegate = delegate

    def plan(self, query: str, config: RagRuntimeConfig, *, enabled: bool) -> HyDEPlan:
        if not enabled:
            return HyDEPlan("disabled", "disabled", "query-transform:none:reason=disabled")
        return HyDEPlan("treatment", None, f"static:v1:{self._delegate.cache_namespace}")

    def execute(self, query: str, plan: HyDEPlan) -> QueryTransformOutcome:
        if plan.cohort != "treatment":
            return NoOpQueryTransformer(reason=plan.bypass_reason or "disabled").transform(query)
        return self._delegate.transform(query)
