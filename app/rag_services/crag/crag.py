"""Corrective-RAG contracts, immutable outcomes, and safe adapters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, Protocol

from app.models.retrieved_chunk import RetrievedChunk

if TYPE_CHECKING:
    from app.rag_services.rag_runtime_config import RagRuntimeConfig

logger = logging.getLogger(__name__)


class CRAGDecision(StrEnum):
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


class GraderParseError(Exception):
    """The grader's structured output was malformed or failed its own
    business-rule validation (e.g. wrong/duplicate chunk indices) after
    exhausting its own retry budget - see
    ``StructuredLLMRetrievalGrader.grade``. Caught specifically by
    ``FailSafeCorrectiveRetriever`` so this failure mode surfaces as
    ``bypass_reason="grader_parse_error"`` instead of being lumped into the
    generic ``"crag_error"`` fallback bucket alongside LLM timeouts, network
    errors, and everything else."""


class EvidenceOrigin(StrEnum):
    POLICY = "policy"
    REGULATORY_WEB = "regulatory_web"


@dataclass(frozen=True)
class ChunkGrade:
    index: int
    relevance: float
    supports_question: bool
    reason_code: str


@dataclass(frozen=True)
class RetrievalGrade:
    decision: CRAGDecision
    coverage: float
    chunks: tuple[ChunkGrade, ...]
    usage_tokens: int = 0
    duration_ms: float = 0.0


@dataclass(frozen=True)
class EvidenceChunk:
    text: str
    source: str
    page_number: int | None
    retrieval_score: float
    origin: EvidenceOrigin
    canonical_url: str | None = None
    retrieved_at_iso: str | None = None


@dataclass(frozen=True)
class WebEvidence:
    title: str
    text: str
    canonical_url: str
    domain: str
    retrieved_at_iso: str
    score: float


@dataclass(frozen=True)
class CRAGOutcome:
    evidence: tuple[EvidenceChunk, ...]
    decision: CRAGDecision | None
    applied: bool
    fallback: bool = False
    abstain: bool = False
    web_used: bool = False
    bypass_reason: str | None = None
    usage_tokens: int = 0
    duration_ms: float = 0.0


class RetrievalGrader(Protocol):
    @property
    def cache_namespace(self) -> str: ...

    def grade(self, question: str, chunks: list[RetrievedChunk]) -> RetrievalGrade: ...


class KnowledgeRefiner(Protocol):
    @property
    def cache_namespace(self) -> str: ...

    def refine_local(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        grade: RetrievalGrade,
    ) -> tuple[tuple[EvidenceChunk, ...], int]: ...

    def refine_web(
        self, question: str, results: list[WebEvidence]
    ) -> tuple[tuple[EvidenceChunk, ...], int]: ...


class SourceScopePolicy(Protocol):
    @property
    def cache_namespace(self) -> str: ...

    def permits_public_regulatory_web(self, question: str) -> bool: ...


class WebRetriever(Protocol):
    @property
    def cache_namespace(self) -> str: ...

    def search(self, query: str) -> list[WebEvidence]: ...


class CorrectiveRetriever(Protocol):
    @property
    def cache_namespace(self) -> str: ...

    def correct(
        self, question: str, chunks: list[RetrievedChunk], *, allow_web: bool
    ) -> CRAGOutcome: ...


CRAGCohort = Literal["disabled", "control", "shadow", "treatment"]


@dataclass(frozen=True)
class CRAGPlan:
    cohort: CRAGCohort
    bypass_reason: str | None
    allow_web: bool
    cache_namespace: str


class PlannedCorrectiveRetriever(Protocol):
    def plan(self, question: str, config: RagRuntimeConfig, *, enabled: bool) -> CRAGPlan: ...

    def execute(
        self, question: str, chunks: list[RetrievedChunk], plan: CRAGPlan
    ) -> CRAGOutcome: ...


def local_evidence(chunks: list[RetrievedChunk]) -> tuple[EvidenceChunk, ...]:
    return tuple(
        EvidenceChunk(
            text=chunk.text,
            source=chunk.source,
            page_number=chunk.page_number,
            retrieval_score=chunk.score,
            origin=EvidenceOrigin.POLICY,
        )
        for chunk in chunks
    )


class PlannedNoOpCorrectiveRetriever:
    def plan(self, question: str, config: RagRuntimeConfig, *, enabled: bool) -> CRAGPlan:
        return CRAGPlan("disabled", "disabled", False, "crag:none:reason=disabled")

    def execute(self, question: str, chunks: list[RetrievedChunk], plan: CRAGPlan) -> CRAGOutcome:
        return CRAGOutcome(
            evidence=local_evidence(chunks),
            decision=None,
            applied=False,
            bypass_reason=plan.bypass_reason or "disabled",
        )


class StaticPlannedCorrectiveRetriever:
    """Eval adapter: enabled means treatment, independent of RAG Ops."""

    def __init__(self, delegate: CorrectiveRetriever) -> None:
        self._delegate = delegate

    def plan(self, question: str, config: RagRuntimeConfig, *, enabled: bool) -> CRAGPlan:
        if not enabled:
            return CRAGPlan("disabled", "disabled", False, "crag:none:reason=disabled")
        return CRAGPlan(
            "treatment", None, False, f"crag:static:v1:{self._delegate.cache_namespace}"
        )

    def execute(self, question: str, chunks: list[RetrievedChunk], plan: CRAGPlan) -> CRAGOutcome:
        if plan.cohort != "treatment":
            return PlannedNoOpCorrectiveRetriever().execute(question, chunks, plan)
        return self._delegate.correct(question, chunks, allow_web=plan.allow_web)


class FailSafeCorrectiveRetriever:
    """Availability boundary around the complete CRAG stage.

    Returning original local evidence is safe for an implementation fault
    only when local chunks exist. With no local evidence, force abstention.
    The caller must not cache either fallback form.
    """

    def __init__(self, delegate: CorrectiveRetriever) -> None:
        self._delegate = delegate

    @property
    def cache_namespace(self) -> str:
        return self._delegate.cache_namespace

    def correct(
        self, question: str, chunks: list[RetrievedChunk], *, allow_web: bool
    ) -> CRAGOutcome:
        try:
            return self._delegate.correct(question, chunks, allow_web=allow_web)
        except GraderParseError as exc:
            logger.warning(
                "rag.crag_fallback",
                extra={"error_type": type(exc).__name__, "bypass_reason": "grader_parse_error"},
            )
            return CRAGOutcome(
                evidence=local_evidence(chunks),
                decision=None,
                applied=False,
                fallback=True,
                abstain=not chunks,
                bypass_reason="grader_parse_error",
            )
        except Exception as exc:  # noqa: BLE001 - deliberate service boundary
            logger.warning(
                "rag.crag_fallback",
                extra={"error_type": type(exc).__name__},
            )
            return CRAGOutcome(
                evidence=local_evidence(chunks),
                decision=None,
                applied=False,
                fallback=True,
                abstain=not chunks,
                bypass_reason="crag_error",
            )
