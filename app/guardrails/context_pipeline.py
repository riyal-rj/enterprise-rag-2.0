"""Context guardrail: scans retrieved/CRAG-corrected evidence for injection
content smuggled inside document text - "ignore previous instructions"
embedded in a policy PDF chunk (or, less likely given
``TavilyRegulatoryWebRetriever``'s own guardrails, a web result) is a
distinct threat from a malicious user prompt, and the input guardrail never
sees it.

A flagged chunk is dropped, not exception-raised: the same "fail closed at
the item level, fail open at the pipeline level" posture
``app.rag_services.crag`` already uses for a single grading failure (see
``FailSafeCorrectiveRetriever``). ``RAGService._build_evidence_context``
already handles an empty evidence tuple gracefully, so dropping every
chunk degrades to "no relevant context was found" rather than crashing.

The scanner passed in here is expected to be injection-focused only (see
``app.guardrails.factory.build_context_guard_pipeline``) - this pipeline
applies it uniformly to every chunk regardless of origin (internal policy
or web-corrected evidence). Web evidence from
``TavilyRegulatoryWebRetriever`` already passed that module's own HTTPS/
domain-allowlist/char-budget/timeout guardrails before reaching here; this
pipeline only adds the injection scan as defense-in-depth on top, not a
second PII/secrets/toxicity pass (nonsensical for regulator text).
"""

from __future__ import annotations

import logging

from app.guardrails.contracts import (
    GuardrailDecision,
    GuardrailMode,
    GuardrailStage,
    ScanFinding,
    TextScanner,
)
from app.guardrails.policy import GuardrailPolicy
from app.guardrails.security_events import summarize_decision
from app.rag_services.crag.crag import EvidenceChunk
from app.repositories.security_events_repository import SecurityEventsRepository

logger = logging.getLogger(__name__)


class ContextGuardPipeline:
    def __init__(
        self,
        *,
        injection_scanner: TextScanner | None,
        policy: GuardrailPolicy,
        security_events: SecurityEventsRepository | None = None,
    ) -> None:
        self._injection_scanner = injection_scanner
        self._policy = policy
        self._security_events = security_events

    def filter_evidence(
        self, evidence: tuple[EvidenceChunk, ...], *, mode: GuardrailMode
    ) -> tuple[EvidenceChunk, ...]:
        if self._injection_scanner is None or not evidence:
            return evidence

        kept: list[EvidenceChunk] = []
        dropped_decisions: list[GuardrailDecision] = []
        for chunk in evidence:
            findings: tuple[ScanFinding, ...] = self._injection_scanner.scan(chunk.text)
            decision = self._policy.decide(findings, stage=GuardrailStage.CONTEXT, mode=mode)
            if decision.blocked:
                dropped_decisions.append(decision)
                continue
            kept.append(chunk)

        if dropped_decisions:
            logger.warning(
                "guardrails.context_chunks_dropped",
                extra={"dropped_count": len(dropped_decisions), "kept_count": len(kept)},
            )
            self._record(dropped_decisions, mode=mode)
        return tuple(kept)

    def _record(self, dropped_decisions: list[GuardrailDecision], *, mode: GuardrailMode) -> None:
        if self._security_events is None:
            return
        # One row per filter_evidence() call, not one per dropped chunk -
        # a document riddled with injection attempts shouldn't drown the
        # audit trail in near-duplicate rows.
        all_findings = tuple(f for d in dropped_decisions for f in d.findings)
        aggregate = GuardrailDecision(
            stage=GuardrailStage.CONTEXT,
            action=dropped_decisions[0].action,
            mode=mode,
            findings=all_findings,
            would_block=any(d.would_block for d in dropped_decisions),
        )
        self._security_events.record(
            actor=None,
            action="context_chunks_dropped",
            stage=GuardrailStage.CONTEXT.value,
            category=(all_findings[0].category.value if all_findings else None),
            mode=mode,
            changes=summarize_decision(aggregate)
            | {"dropped_chunk_count": len(dropped_decisions)},
        )


class NoOpContextGuardPipeline(ContextGuardPipeline):
    """Pass-through default - see ``app.guardrails.input_pipeline.NoOpInputGuardPipeline``."""

    def __init__(self) -> None:
        super().__init__(injection_scanner=None, policy=GuardrailPolicy())

    def filter_evidence(
        self, evidence: tuple[EvidenceChunk, ...], *, mode: GuardrailMode = "enforce"
    ) -> tuple[EvidenceChunk, ...]:
        del mode
        return evidence
