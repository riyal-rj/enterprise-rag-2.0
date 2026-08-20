"""Pure decision function: findings + mode -> a :class:`GuardrailDecision`.

No I/O, no scanner dependency - independently unit-testable with hand-built
``ScanFinding`` tuples. Mirrors the role
``app.rag_services.reflection.ThresholdReflectionDecisionPolicy`` plays for
self-reflection: the scanning/generation happens elsewhere, this module only
turns evidence into a decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.guardrails.contracts import (
    GuardrailAction,
    GuardrailCategory,
    GuardrailDecision,
    GuardrailMode,
    GuardrailStage,
    ScanFinding,
)

# Categories that are always block-worthy once *any* finding names them,
# regardless of score - an injection or a leaked secret has no legitimate
# "redact and continue" story the way PII sometimes does.
_ALWAYS_BLOCK: frozenset[GuardrailCategory] = frozenset(
    {
        GuardrailCategory.PROMPT_INJECTION,
        GuardrailCategory.SECRETS,
        GuardrailCategory.MALICIOUS_URL,
        GuardrailCategory.PROMPT_LEAK,
    }
)

# Categories that are block-worthy only above a per-category score
# threshold - a low-confidence toxicity/banned-topic hit is noisier than a
# structural pattern match, so it gets a threshold instead of an automatic
# block.
_BLOCK_ABOVE_THRESHOLD: dict[GuardrailCategory, float] = {
    GuardrailCategory.TOXICITY: 0.75,
    GuardrailCategory.BANNED_TOPIC: 0.6,
}

# PII is redact-worthy, not block-worthy: in a banking context a user
# legitimately pastes an account number, and refusing the whole request
# would be worse than masking it and continuing. UNSAFE_REFUSAL is
# monitoring-only (see llm_guard_adapter's NoRefusal note) - it never drives
# an action on its own.
_REDACT_ONLY: frozenset[GuardrailCategory] = frozenset({GuardrailCategory.PII})


@dataclass(frozen=True)
class GuardrailPolicy:
    """Per-category severity mapping, overridable per instance for tests -
    the module-level defaults above back the zero-arg constructor."""

    always_block: frozenset[GuardrailCategory] = field(default_factory=lambda: _ALWAYS_BLOCK)
    block_above_threshold: dict[GuardrailCategory, float] = field(
        default_factory=lambda: dict(_BLOCK_ABOVE_THRESHOLD)
    )
    redact_only: frozenset[GuardrailCategory] = field(default_factory=lambda: _REDACT_ONLY)

    def decide(
        self,
        findings: tuple[ScanFinding, ...],
        *,
        stage: GuardrailStage,
        mode: GuardrailMode,
    ) -> GuardrailDecision:
        would_block = any(self._is_block_worthy(f) for f in findings)
        would_redact = not would_block and any(f.category in self.redact_only for f in findings)

        if mode == "monitor":
            # Monitor mode observes but never applies BLOCK/REDACT - the
            # would-be action is preserved in `would_block` so
            # security_events can record "this would have been blocked"
            # without it ever affecting what's served.
            action = GuardrailAction.ALLOW
        elif would_block:
            action = GuardrailAction.BLOCK
        elif would_redact:
            action = GuardrailAction.REDACT
        else:
            action = GuardrailAction.ALLOW

        return GuardrailDecision(
            stage=stage,
            action=action,
            mode=mode,
            findings=findings,
            would_block=would_block,
        )

    def _is_block_worthy(self, finding: ScanFinding) -> bool:
        if finding.category in self.always_block:
            return True
        threshold = self.block_above_threshold.get(finding.category)
        return threshold is not None and finding.score >= threshold
