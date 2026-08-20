"""Input guardrail: canonicalize -> deterministic scanners -> ML scanner ->
policy decision. Runs unconditionally ahead of intent routing (see
``app.query_orchestration.query_orchestrator.QueryOrchestrator._guard_input``)
- no rollout-percentage gating, since guardrails apply to every request
regardless of which pipeline (RAG/SQL/hybrid) ends up handling it.
"""

from __future__ import annotations

from app.guardrails.canonicalizer import canonicalize
from app.guardrails.contracts import (
    GuardrailAction,
    GuardrailBlockedError,
    GuardrailDecision,
    GuardrailMode,
    GuardrailStage,
    ScanFinding,
    TextScanner,
)
from app.guardrails.policy import GuardrailPolicy
from app.guardrails.security_events import summarize_decision
from app.guardrails.sensitive_data import SensitiveValueVault, tokenize
from app.repositories.security_events_repository import SecurityEventsRepository


class InputGuardPipeline:
    def __init__(
        self,
        *,
        deterministic_scanners: tuple[TextScanner, ...],
        ml_scanner: TextScanner | None,
        policy: GuardrailPolicy,
        security_events: SecurityEventsRepository | None = None,
    ) -> None:
        self._deterministic_scanners = deterministic_scanners
        self._ml_scanner = ml_scanner
        self._policy = policy
        self._security_events = security_events

    def check(self, text: str, *, mode: GuardrailMode, actor: str | None = None) -> str:
        """Returns the sanitized text on ALLOW/REDACT, raises
        :class:`GuardrailBlockedError` on BLOCK. Thin wrapper around
        :meth:`check_with_decision` for callers that don't need the
        decision itself."""
        sanitized, _decision = self.check_with_decision(text, mode=mode, actor=actor)
        return sanitized

    def check_with_decision(
        self, text: str, *, mode: GuardrailMode, actor: str | None = None
    ) -> tuple[str, GuardrailDecision]:
        """Same as :meth:`check`, but also returns the
        :class:`GuardrailDecision` - callers that need to reflect the real
        action taken (e.g. a REDACT) in response metadata further downstream
        (see ``QueryOrchestrator.answer``, which can't otherwise tell a
        redacted-but-allowed question from a clean one once ``RAGService``
        builds its own metadata with no knowledge of this stage) use this
        instead of :meth:`check`. ``actor`` is the calling principal's
        username, if known - purely for the sanitized ``security_events``
        audit row, never logged/used for anything else."""
        canonicalized = canonicalize(text)

        findings: list[ScanFinding] = []
        for scanner in self._deterministic_scanners:
            findings.extend(scanner.scan(canonicalized))
        if self._ml_scanner is not None:
            findings.extend(self._ml_scanner.scan(canonicalized))

        decision = self._policy.decide(
            tuple(findings), stage=GuardrailStage.INPUT, mode=mode
        )
        if decision.would_block or decision.action.value == "redact":
            self._record(decision, actor=actor)
        if decision.blocked:
            blocking_category = next(
                (f.category for f in decision.findings), None
            )
            raise GuardrailBlockedError(stage=GuardrailStage.INPUT, category=blocking_category)

        if decision.action.value == "redact":
            vault = SensitiveValueVault()
            return tokenize(canonicalized, vault), decision
        return canonicalized, decision

    def _record(self, decision: GuardrailDecision, actor: str | None) -> None:
        if self._security_events is None:
            return
        # In monitor mode a block-worthy finding never becomes an applied
        # BLOCK (see GuardrailPolicy.decide), so the action name reflects
        # that distinctly - "would have blocked", not "blocked" - rather
        # than misleadingly implying enforcement happened.
        action_name = "input_would_block" if decision.action.value == "allow" else (
            f"input_{decision.action.value}"
        )
        self._security_events.record(
            actor=actor,
            action=action_name,
            stage=GuardrailStage.INPUT.value,
            category=(decision.findings[0].category.value if decision.findings else None),
            mode=decision.mode,
            changes=summarize_decision(decision),
        )


class NoOpInputGuardPipeline(InputGuardPipeline):
    """Pass-through used as the safe default before real scanners are wired
    in (see ``app.guardrails.factory``) and by tests that don't care about
    guardrail behavior."""

    def __init__(self) -> None:
        super().__init__(deterministic_scanners=(), ml_scanner=None, policy=GuardrailPolicy())

    def check(self, text: str, *, mode: GuardrailMode = "enforce", actor: str | None = None) -> str:
        del mode, actor
        return text

    def check_with_decision(
        self, text: str, *, mode: GuardrailMode = "enforce", actor: str | None = None
    ) -> tuple[str, GuardrailDecision]:
        del actor
        return text, GuardrailDecision(
            stage=GuardrailStage.INPUT, action=GuardrailAction.ALLOW, mode=mode
        )
