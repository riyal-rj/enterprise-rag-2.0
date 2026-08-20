"""Self-reflective answer contracts and immutable request outcomes.

A Self-RAG-*inspired* production adaptation, not a claim of reproducing the
official Self-RAG paper's trained reflection-token model: this maps the same
four judgments (is retrieval needed, is evidence relevant, is the answer
supported, is the answer useful) onto strict structured fields, and lets
application code - not the model - deterministically pick
accept/revise/retrieve-more/abstain. See ``StructuredSelfReflectionEngine``
for the bounded loop and ``ThresholdReflectionDecisionPolicy`` for the
deterministic policy.

Placement in the pipeline mirrors ``app.rag_services.crag``: CRAG asks
whether the *evidence* is good enough; this stage asks whether the
*generated answer* used that evidence correctly and completely. It runs
after CRAG's evidence correction and the initial answer generation, sees
only CRAG-approved real evidence (never a HyDE hypothesis), and is bounded
by iteration/retrieval/token/deadline budgets - never a recursive call back
into ``RAGService.answer()``.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, Protocol

from app.rag_services.crag import EvidenceChunk

if TYPE_CHECKING:
    from app.rag_services.rag_runtime_config import RagRuntimeConfig

logger = logging.getLogger(__name__)


class SupportLevel(StrEnum):
    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"


class ReflectionAction(StrEnum):
    ACCEPT = "accept"
    REVISE = "revise"
    RETRIEVE_MORE = "retrieve_more"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class ReflectionCritique:
    retrieval_needed: bool
    retrieval_query: str | None
    evidence_relevance: float
    support_level: SupportLevel
    answer_relevance: float
    citation_completeness: float
    utility: int
    missing_aspects: tuple[str, ...]
    unsupported_claims: tuple[str, ...]
    usage_tokens: int = 0
    duration_ms: float = 0.0


@dataclass(frozen=True)
class ReflectionBudget:
    max_iterations: int
    max_additional_retrievals: int
    max_total_tokens: int
    deadline_monotonic: float


@dataclass(frozen=True)
class ReflectionState:
    answer: str
    evidence: tuple[EvidenceChunk, ...]
    iteration: int = 0
    additional_retrievals: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class SelfReflectionOutcome:
    answer: str
    evidence: tuple[EvidenceChunk, ...]
    applied: bool
    accepted: bool
    final_action: ReflectionAction
    iterations: int
    additional_retrievals: int
    fallback: bool = False
    abstain: bool = False
    bypass_reason: str | None = None
    support_level: SupportLevel | None = None
    answer_relevance: float | None = None
    citation_completeness: float | None = None
    utility: int | None = None
    usage_tokens: int = 0
    duration_ms: float = 0.0


class ReflectionCritic(Protocol):
    @property
    def cache_namespace(self) -> str: ...

    def critique(
        self,
        question: str,
        evidence: tuple[EvidenceChunk, ...],
        answer: str,
        *,
        timeout_seconds: float | None = None,
    ) -> ReflectionCritique: ...


class ReflectionDecisionPolicy(Protocol):
    @property
    def cache_namespace(self) -> str: ...

    def decide(
        self,
        critique: ReflectionCritique,
        state: ReflectionState,
        budget: ReflectionBudget,
        *,
        deterministic_unsupported_claims: tuple[str, ...],
    ) -> ReflectionAction: ...


class GroundedAnswerReviser(Protocol):
    @property
    def cache_namespace(self) -> str: ...

    def revise(
        self,
        question: str,
        evidence: tuple[EvidenceChunk, ...],
        previous_answer: str,
        critique: ReflectionCritique,
        *,
        timeout_seconds: float | None = None,
    ) -> tuple[str, int]: ...


class EvidenceAugmenter(Protocol):
    def retrieve(self, query: str) -> tuple[EvidenceChunk, ...]: ...


class SelfReflectionEngine(Protocol):
    @property
    def cache_namespace(self) -> str: ...

    def reflect(
        self,
        question: str,
        evidence: tuple[EvidenceChunk, ...],
        initial_answer: str,
        augmenter: EvidenceAugmenter,
        *,
        allow_retrieval: bool = True,
    ) -> SelfReflectionOutcome: ...


SelfReflectionCohort = Literal["disabled", "control", "shadow", "treatment"]


@dataclass(frozen=True)
class SelfReflectionPlan:
    cohort: SelfReflectionCohort
    bypass_reason: str | None
    cache_namespace: str
    # Whether the bounded additional-retrieval step (RETRIEVE_MORE) is
    # available for this request - the revision-only-canary gate, same role
    # CRAGPlan.allow_web plays for CRAG's web correction. Always False for
    # every non-treatment cohort (mirrors allow_web always being False in
    # shadow/control/disabled - see DynamicSelfReflectionEngine.plan).
    allow_retrieval: bool = False


class PlannedSelfReflectionEngine(Protocol):
    def plan(
        self, question: str, config: RagRuntimeConfig, *, enabled: bool
    ) -> SelfReflectionPlan: ...

    def execute(
        self,
        question: str,
        evidence: tuple[EvidenceChunk, ...],
        initial_answer: str,
        augmenter: EvidenceAugmenter,
        plan: SelfReflectionPlan,
    ) -> SelfReflectionOutcome: ...


class PlannedNoOpSelfReflectionEngine:
    def plan(self, question: str, config: RagRuntimeConfig, *, enabled: bool) -> SelfReflectionPlan:
        return SelfReflectionPlan("disabled", "disabled", "self-reflection:none:reason=disabled")

    def execute(
        self,
        question: str,
        evidence: tuple[EvidenceChunk, ...],
        initial_answer: str,
        augmenter: EvidenceAugmenter,
        plan: SelfReflectionPlan,
    ) -> SelfReflectionOutcome:
        return SelfReflectionOutcome(
            answer=initial_answer,
            evidence=evidence,
            applied=False,
            accepted=True,
            final_action=ReflectionAction.ACCEPT,
            iterations=0,
            additional_retrievals=0,
            bypass_reason=plan.bypass_reason or "disabled",
        )


class FailSafeSelfReflectionEngine:
    """Availability boundary around the complete reflection stage.

    Any failure (LLM error, schema validation, retrieval error inside the
    bounded evidence augmentation) degrades to the untouched initial answer
    - never a partially-revised one - marked ``fallback=True,
    accepted=False``. The caller must never cache this outcome (see
    ``RAGService.answer``'s ``pipeline_fallback`` computation).
    """

    def __init__(self, delegate: SelfReflectionEngine) -> None:
        self._delegate = delegate

    @property
    def cache_namespace(self) -> str:
        return self._delegate.cache_namespace

    def reflect(
        self,
        question: str,
        evidence: tuple[EvidenceChunk, ...],
        initial_answer: str,
        augmenter: EvidenceAugmenter,
        *,
        allow_retrieval: bool = True,
    ) -> SelfReflectionOutcome:
        try:
            return self._delegate.reflect(
                question, evidence, initial_answer, augmenter, allow_retrieval=allow_retrieval
            )
        except Exception as exc:  # noqa: BLE001 - deliberate availability boundary
            logger.warning(
                "rag.self_reflection_fallback",
                extra={"error_type": type(exc).__name__},
            )
            return SelfReflectionOutcome(
                answer=initial_answer,
                evidence=evidence,
                applied=False,
                accepted=False,
                final_action=ReflectionAction.ABSTAIN,
                iterations=0,
                additional_retrievals=0,
                fallback=True,
                abstain=False,
                bypass_reason="reflection_error",
            )


_MIN_QUERY_LENGTH = 3
_MAX_QUERY_LENGTH = 500
_FORBIDDEN_QUERY_MARKERS = (
    "ignore previous",
    "ignore all previous",
    "system prompt",
    "http://",
    "https://",
    "sql:",
)


def validate_reflection_query(value: str) -> str:
    """Sanity/safety check for a critic-generated retrieval query before it
    reaches ``EvidenceAugmenter.retrieve`` - the query is untrusted, model-
    generated text, not a vetted user query. Rejects instruction-like,
    URL-bearing, or control-character-bearing content rather than trying to
    sanitize it; the caller (the reflection engine) treats a ``ValueError``
    here the same as any other reflection-stage failure - the outer
    ``FailSafeSelfReflectionEngine`` catches it."""
    if any(unicodedata.category(ch) == "Cc" for ch in value):
        # Rejected before whitespace normalization, which only collapses
        # str.split()'s notion of whitespace - most control characters
        # (NUL, ESC, ...) aren't whitespace and would otherwise survive
        # into a query handed to retrieval/embedding/logging untouched.
        raise ValueError("reflection retrieval query contains a control character")
    normalized = " ".join(value.split())
    if not _MIN_QUERY_LENGTH <= len(normalized) <= _MAX_QUERY_LENGTH:
        raise ValueError("reflection retrieval query length is invalid")
    lowered = normalized.casefold()
    if any(marker in lowered for marker in _FORBIDDEN_QUERY_MARKERS):
        raise ValueError("reflection retrieval query contains a forbidden marker")
    return normalized
