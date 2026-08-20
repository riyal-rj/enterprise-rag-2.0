"""Output guardrail: scans the final answer text before it can be cached or
returned to a caller - see the two insertion points in ``RAGService.answer``
(``app.rag_services.rag_service``): right after ``ChatResponse`` is built,
before the ``pipeline_fallback``-gated cache writes.

On BLOCK, the caller is expected to substitute a fixed safe-abstention
string rather than propagate the exception to the HTTP layer - an answer
that already cost an LLM call shouldn't 500 (same posture
``_SELF_REFLECTION_FALLBACK_ABSTENTION`` already uses in that module).
"""

from __future__ import annotations

from app.guardrails.contracts import (
    GuardrailBlockedError,
    GuardrailDecision,
    GuardrailMode,
    GuardrailStage,
    OutputScanner,
    ScanFinding,
    TextScanner,
)
from app.guardrails.policy import GuardrailPolicy
from app.guardrails.security_events import summarize_decision
from app.guardrails.sensitive_data import SensitiveValueVault, tokenize
from app.repositories.security_events_repository import SecurityEventsRepository


class OutputGuardPipeline:
    def __init__(
        self,
        *,
        deterministic_scanners: tuple[TextScanner, ...],
        ml_scanner: OutputScanner | None,
        policy: GuardrailPolicy,
        security_events: SecurityEventsRepository | None = None,
    ) -> None:
        self._deterministic_scanners = deterministic_scanners
        self._ml_scanner = ml_scanner
        self._policy = policy
        self._security_events = security_events

    def apply(self, *, prompt: str, answer: str, mode: GuardrailMode) -> str:
        """Returns the (possibly redacted) answer on ALLOW/REDACT, raises
        :class:`GuardrailBlockedError` on BLOCK."""
        findings: list[ScanFinding] = []
        for scanner in self._deterministic_scanners:
            findings.extend(scanner.scan(answer))
        if self._ml_scanner is not None:
            findings.extend(self._ml_scanner.scan(prompt=prompt, output=answer))

        decision = self._policy.decide(
            tuple(findings), stage=GuardrailStage.OUTPUT, mode=mode
        )
        if decision.would_block or decision.action.value == "redact":
            self._record(decision)
        if decision.blocked:
            blocking_category = next((f.category for f in decision.findings), None)
            raise GuardrailBlockedError(stage=GuardrailStage.OUTPUT, category=blocking_category)

        if decision.action.value == "redact":
            return tokenize(answer, SensitiveValueVault())
        return answer

    def _record(self, decision: GuardrailDecision) -> None:
        if self._security_events is None:
            return
        action_name = "output_would_block" if decision.action.value == "allow" else (
            f"output_{decision.action.value}"
        )
        self._security_events.record(
            actor=None,
            action=action_name,
            stage=GuardrailStage.OUTPUT.value,
            category=(decision.findings[0].category.value if decision.findings else None),
            mode=decision.mode,
            changes=summarize_decision(decision),
        )


class NoOpOutputGuardPipeline(OutputGuardPipeline):
    """Pass-through default - see ``app.guardrails.input_pipeline.NoOpInputGuardPipeline``."""

    def __init__(self) -> None:
        super().__init__(deterministic_scanners=(), ml_scanner=None, policy=GuardrailPolicy())

    def apply(self, *, prompt: str, answer: str, mode: GuardrailMode = "enforce") -> str:
        del prompt, mode
        return answer
