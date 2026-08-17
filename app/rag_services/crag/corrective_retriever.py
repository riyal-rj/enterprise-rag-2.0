"""Correct/ambiguous/incorrect CRAG orchestration."""

from __future__ import annotations

import time

from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.crag.crag import (
    CRAGDecision,
    CRAGOutcome,
    KnowledgeRefiner,
    RetrievalGrader,
    SourceScopePolicy,
    WebRetriever,
)


class ProductionCorrectiveRetriever:
    def __init__(
        self,
        *,
        grader: RetrievalGrader,
        refiner: KnowledgeRefiner,
        scope_policy: SourceScopePolicy,
        web_retriever: WebRetriever | None,
        min_evidence_chunks: int = 1,
    ) -> None:
        self._grader = grader
        self._refiner = refiner
        self._scope = scope_policy
        self._web = web_retriever
        self._min_evidence = min_evidence_chunks

    @property
    def cache_namespace(self) -> str:
        web = self._web.cache_namespace if self._web is not None else "none"
        return (
            f"crag:v1:{self._grader.cache_namespace}:"
            f"{self._refiner.cache_namespace}:web={web}:"
            f"min_evidence={self._min_evidence}"
        )

    def correct(
        self, question: str, chunks: list[RetrievedChunk], *, allow_web: bool
    ) -> CRAGOutcome:
        started = time.perf_counter()
        grade = self._grader.grade(question, chunks)
        local, refinement_tokens = self._refiner.refine_local(question, chunks, grade)
        tokens = grade.usage_tokens + refinement_tokens

        if grade.decision is CRAGDecision.CORRECT and len(local) >= self._min_evidence:
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
            results = self._web.search(question)
            web_evidence, web_tokens = self._refiner.refine_web(question, results)
            tokens += web_tokens
            combined = tuple((*local, *web_evidence))
            return CRAGOutcome(
                evidence=combined,
                decision=grade.decision,
                applied=True,
                web_used=bool(web_evidence),
                abstain=len(combined) < self._min_evidence,
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
