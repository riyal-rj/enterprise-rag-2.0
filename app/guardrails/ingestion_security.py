"""Ingestion-time scanning: the "scan" step of the policy-upload quarantine
state machine (``pending_scan`` -> ``scan_passed``/``scan_failed`` -> ...,
see ``app.services.policy_ingestion_security_service`` and
``app/seed/migrations/014_create_document_security_state.sql``).

A malicious/poisoned document is exactly a "poisoned evidence chunk" threat
at ingestion time rather than query time, so this reuses the same
injection/secrets/PII scanners the context and input guards use, over the
concatenated extracted document text.
"""

from __future__ import annotations

from app.guardrails.contracts import (
    GuardrailAction,
    GuardrailDecision,
    GuardrailMode,
    GuardrailStage,
    ScanFinding,
    TextScanner,
)
from app.guardrails.policy import GuardrailPolicy
from app.guardrails.security_events import summarize_decision
from app.repositories.security_events_repository import SecurityEventsRepository


class IngestionSecurityScanner:
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

    def scan_document(
        self, chunk_texts: list[str], *, mode: GuardrailMode, actor: str | None = None
    ) -> GuardrailDecision:
        """Scans every chunk's text and aggregates findings across the whole
        document. REDACT is treated as ALLOW-with-warning here (masking a
        whole document isn't meaningful the way per-response redaction is);
        the findings are still surfaced to the approving admin via the
        decision's ``findings``. ``actor`` is the uploading admin's
        username, if known (see
        ``PolicyIngestionSecurityService.submit``) - for the sanitized
        ``security_events`` audit row only.
        """
        findings: list[ScanFinding] = []
        for text in chunk_texts:
            for scanner in self._deterministic_scanners:
                findings.extend(scanner.scan(text))
            if self._ml_scanner is not None:
                findings.extend(self._ml_scanner.scan(text))

        decision = self._policy.decide(
            tuple(findings), stage=GuardrailStage.INGESTION, mode=mode
        )
        if decision.would_block:
            self._record(decision, actor=actor)
        if decision.action is GuardrailAction.REDACT:
            decision = GuardrailDecision(
                stage=GuardrailStage.INGESTION,
                action=GuardrailAction.ALLOW,
                mode=decision.mode,
                findings=decision.findings,
                would_block=decision.would_block,
            )
        return decision

    def _record(self, decision: GuardrailDecision, *, actor: str | None) -> None:
        if self._security_events is None:
            return
        action_name = (
            "ingestion_would_block" if decision.action is GuardrailAction.ALLOW
            else "ingestion_scan_failed"
        )
        self._security_events.record(
            actor=actor,
            action=action_name,
            stage=GuardrailStage.INGESTION.value,
            category=(decision.findings[0].category.value if decision.findings else None),
            mode=decision.mode,
            changes=summarize_decision(decision),
        )
