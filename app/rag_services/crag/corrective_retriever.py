"""Correct/ambiguous/incorrect CRAG orchestration."""

from __future__ import annotations

import logging
import time

from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.crag.crag import (
    CRAGDecision,
    CRAGOutcome,
    EvidenceChunk,
    KnowledgeRefiner,
    RetrievalGrade,
    RetrievalGrader,
    SourceScopePolicy,
    WebQueryFormulator,
    WebRetriever,
)

logger = logging.getLogger(__name__)


def _log_chunk_grading(chunks: list[RetrievedChunk], grade: RetrievalGrade) -> None:
    """Per-chunk grading trail: initial retrieval score, source/page, the
    grader's relevance/support/reason-code decision for every graded chunk,
    and whether it cleared the bar `refine_local` extracts evidence from
    (``supports_question and relevance >= min_relevance`` - the exact
    predicate `refine_local` applies, duplicated here only for visibility,
    never as a second source of truth). Deliberately omits chunk text and
    the question - source/page/scores/reason codes are enough to diagnose a
    bad grade without logging retrieval content (matching the CRAGMetadata
    API contract's same restriction, see ``app.schemas.chat``)."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    graded_by_index = {item.index: item for item in grade.chunks}
    rows = [
        {
            "index": index,
            "source": chunk.source,
            "page_number": chunk.page_number,
            "retrieval_score": round(chunk.score, 4),
            "graded": index in graded_by_index,
            "grader_relevance": (
                graded_by_index[index].relevance if index in graded_by_index else None
            ),
            "supports_question": (
                graded_by_index[index].supports_question if index in graded_by_index else None
            ),
            "reason_code": (
                graded_by_index[index].reason_code if index in graded_by_index else None
            ),
        }
        for index, chunk in enumerate(chunks)
    ]
    logger.debug(
        "rag.crag_chunk_grading",
        extra={
            "coverage": grade.coverage,
            "overall_decision": grade.decision.value,
            "chunks": rows,
        },
    )


def _evidence_as_retrieved(evidence: tuple[EvidenceChunk, ...]) -> list[RetrievedChunk]:
    """Reshape already-approved evidence back into ``RetrievedChunk`` so the
    post-correction grade below can regrade combined local+web evidence
    through the same ``RetrievalGrader`` contract used for the initial
    grade."""
    return [
        RetrievedChunk(
            text=item.text,
            source=item.canonical_url or item.source,
            score=item.retrieval_score,
            page_number=item.page_number,
        )
        for item in evidence
    ]


class ProductionCorrectiveRetriever:
    def __init__(
        self,
        *,
        grader: RetrievalGrader,
        refiner: KnowledgeRefiner,
        scope_policy: SourceScopePolicy,
        web_retriever: WebRetriever | None,
        web_query_formulator: WebQueryFormulator | None = None,
        min_evidence_chunks: int = 1,
    ) -> None:
        self._grader = grader
        self._refiner = refiner
        self._scope = scope_policy
        self._web = web_retriever
        # None-able independently of web_retriever: a caller with no
        # formulator just searches with the raw question, same as before
        # this existed - see web_query_formulator.py's module docstring for
        # why a formulator is worth having at all.
        self._query_formulator = web_query_formulator
        self._min_evidence = min_evidence_chunks

    @property
    def cache_namespace(self) -> str:
        web = self._web.cache_namespace if self._web is not None else "none"
        query_formulator = (
            self._query_formulator.cache_namespace if self._query_formulator is not None else "none"
        )
        return (
            f"crag:v1:{self._grader.cache_namespace}:"
            f"{self._refiner.cache_namespace}:"
            f"{self._scope.cache_namespace}:"
            f"web={web}:"
            f"query_formulator={query_formulator}:"
            f"min_evidence={self._min_evidence}"
        )

    def correct(
        self, question: str, chunks: list[RetrievedChunk], *, allow_web: bool
    ) -> CRAGOutcome:
        started = time.perf_counter()
        grade = self._grader.grade(question, chunks)
        local, refinement_tokens = self._refiner.refine_local(question, chunks, grade)
        tokens = grade.usage_tokens + refinement_tokens
        _log_chunk_grading(chunks, grade)

        # Gate on extracted evidence, not on the overall coverage-based
        # decision. `grade.decision` answers "does the retrieved set fully
        # cover every material part of the question" (CORRECT requires
        # coverage >= correct_threshold) - a multi-part question answered by
        # one genuinely relevant, on-point document (e.g. "does overdue
        # re-KYC freeze every transaction" answered by the re-KYC policy's
        # debit-restriction clause, with nothing retrieved about credits)
        # lands AMBIGUOUS on coverage alone even though `refine_local`
        # already extracted real, grader-approved supporting evidence for
        # it. Discarding that evidence and abstaining just because the
        # question wasn't *completely* covered was the bug: it silently
        # threw away correct retrieval + correct grading. `local` is only
        # ever non-empty when at least one chunk cleared
        # `supports_question and relevance >= min_relevance` (see
        # `refine_local`), which by construction can't happen when
        # `grade.decision is INCORRECT` (INCORRECT requires zero such
        # chunks) - so this check is exactly "is there enough real,
        # extracted, approved evidence to answer from", independent of
        # whether that evidence happens to cover the whole question. The
        # answer LLM's system prompt already distinguishes "mandatory now"
        # from "not established by the supplied context" per claim, so
        # partial coverage is surfaced there, not manufactured here.
        if len(local) >= self._min_evidence:
            return CRAGOutcome(
                evidence=local,
                decision=grade.decision,
                applied=True,
                usage_tokens=tokens,
                duration_ms=(time.perf_counter() - started) * 1_000,
            )

        web_allowed = (
            allow_web
            and self._web is not None
            and self._scope.permits_public_regulatory_web(question)
        )
        if web_allowed and self._web is not None:
            search_query = question
            if self._query_formulator is not None:
                search_query, formulation_tokens = self._query_formulator.formulate(question)
                tokens += formulation_tokens
            results = self._web.search(search_query)
            # Evidence selection still runs against the original question,
            # not the abbreviated search query - only the Tavily call itself
            # benefits from keyword-shaped input (see
            # web_query_formulator.py's module docstring).
            web_evidence, web_tokens = self._refiner.refine_web(question, results)
            tokens += web_tokens
            if not web_evidence:
                # No web results survived search+refinement - local evidence
                # alone already failed the CORRECT bar above, so there is
                # nothing new to re-grade. Abstain rather than let one
                # ambiguous local chunk plus zero web evidence look like a
                # resolved question.
                return CRAGOutcome(
                    evidence=local,
                    decision=grade.decision,
                    applied=True,
                    abstain=True,
                    web_used=False,
                    bypass_reason="web_correction_empty",
                    usage_tokens=tokens,
                    duration_ms=(time.perf_counter() - started) * 1_000,
                )

            combined = tuple((*local, *web_evidence))
            # Re-grade the corrected evidence set - "at least one chunk
            # exists" is not the same as "the question is answerable"; only
            # a CORRECT re-grade with enough combined evidence counts as
            # resolved.
            verification = self._grader.grade(question, _evidence_as_retrieved(combined))
            tokens += verification.usage_tokens
            corrected_is_sufficient = (
                verification.decision is CRAGDecision.CORRECT
                and len(combined) >= self._min_evidence
            )
            return CRAGOutcome(
                evidence=combined,
                decision=grade.decision,
                applied=True,
                web_used=True,
                abstain=not corrected_is_sufficient,
                bypass_reason=(
                    None if corrected_is_sufficient else "corrected_evidence_insufficient"
                ),
                usage_tokens=tokens,
                duration_ms=(time.perf_counter() - started) * 1_000,
            )

        # For internal-policy questions, relevant local fragments may be
        # returned as evidence only with an explicit insufficient-evidence
        # instruction. The answer path must abstain rather than fill gaps.
        return CRAGOutcome(
            evidence=local,
            decision=grade.decision,
            applied=True,
            abstain=True,
            bypass_reason="external_correction_not_permitted",
            usage_tokens=tokens,
            duration_ms=(time.perf_counter() - started) * 1_000,
        )
