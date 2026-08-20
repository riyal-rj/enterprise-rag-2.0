"""GuardrailPolicy.decide is a pure function - hand-built findings, no
scanners at all. This is the truth table that defines what "enforce" and
"monitor" actually mean."""

from __future__ import annotations

import pytest

from app.guardrails.contracts import (
    GuardrailAction,
    GuardrailCategory,
    GuardrailStage,
    ScanFinding,
)
from app.guardrails.policy import GuardrailPolicy


def _finding(category: GuardrailCategory, score: float = 1.0) -> ScanFinding:
    return ScanFinding(category=category, score=score, detector="test")


@pytest.mark.parametrize(
    "category",
    [
        GuardrailCategory.PROMPT_INJECTION,
        GuardrailCategory.SECRETS,
        GuardrailCategory.MALICIOUS_URL,
        GuardrailCategory.PROMPT_LEAK,
    ],
)
def test_always_block_categories_block_regardless_of_score(
    category: GuardrailCategory,
) -> None:
    decision = GuardrailPolicy().decide(
        (_finding(category, score=0.01),), stage=GuardrailStage.INPUT, mode="enforce"
    )

    assert decision.action is GuardrailAction.BLOCK
    assert decision.blocked is True


def test_toxicity_below_threshold_is_allowed() -> None:
    decision = GuardrailPolicy().decide(
        (_finding(GuardrailCategory.TOXICITY, score=0.5),),
        stage=GuardrailStage.INPUT,
        mode="enforce",
    )

    assert decision.action is GuardrailAction.ALLOW


def test_toxicity_at_or_above_threshold_blocks() -> None:
    decision = GuardrailPolicy().decide(
        (_finding(GuardrailCategory.TOXICITY, score=0.75),),
        stage=GuardrailStage.INPUT,
        mode="enforce",
    )

    assert decision.action is GuardrailAction.BLOCK


def test_pii_redacts_rather_than_blocks() -> None:
    """In a banking context a user legitimately pastes an account number -
    masking it and continuing is the right answer, refusing the whole
    request is not."""
    decision = GuardrailPolicy().decide(
        (_finding(GuardrailCategory.PII),), stage=GuardrailStage.INPUT, mode="enforce"
    )

    assert decision.action is GuardrailAction.REDACT
    assert decision.blocked is False


def test_a_blocking_finding_wins_over_a_redact_only_finding() -> None:
    decision = GuardrailPolicy().decide(
        (_finding(GuardrailCategory.PII), _finding(GuardrailCategory.PROMPT_INJECTION)),
        stage=GuardrailStage.INPUT,
        mode="enforce",
    )

    assert decision.action is GuardrailAction.BLOCK


def test_unsafe_refusal_is_monitoring_only_and_never_drives_an_action() -> None:
    """A grounded abstention ("the evidence is insufficient") is a
    legitimate banking-RAG answer, so NoRefusal's finding is recorded but
    must never block or redact."""
    decision = GuardrailPolicy().decide(
        (_finding(GuardrailCategory.UNSAFE_REFUSAL),),
        stage=GuardrailStage.OUTPUT,
        mode="enforce",
    )

    assert decision.action is GuardrailAction.ALLOW
    assert decision.would_block is False


def test_no_findings_allows() -> None:
    decision = GuardrailPolicy().decide((), stage=GuardrailStage.INPUT, mode="enforce")

    assert decision.action is GuardrailAction.ALLOW
    assert decision.findings == ()


def test_monitor_mode_downgrades_a_block_to_allow_but_records_would_block() -> None:
    """The whole point of monitor mode: measure what enforcement *would*
    have done against real traffic without ever affecting a response."""
    findings = (_finding(GuardrailCategory.PROMPT_INJECTION),)

    decision = GuardrailPolicy().decide(
        findings, stage=GuardrailStage.INPUT, mode="monitor"
    )

    assert decision.action is GuardrailAction.ALLOW
    assert decision.blocked is False
    assert decision.would_block is True
    assert decision.findings == findings


def test_monitor_mode_downgrades_a_redact_to_allow() -> None:
    decision = GuardrailPolicy().decide(
        (_finding(GuardrailCategory.PII),), stage=GuardrailStage.INPUT, mode="monitor"
    )

    assert decision.action is GuardrailAction.ALLOW


def test_decision_echoes_back_its_stage_and_mode() -> None:
    decision = GuardrailPolicy().decide(
        (), stage=GuardrailStage.CONTEXT, mode="monitor"
    )

    assert decision.stage is GuardrailStage.CONTEXT
    assert decision.mode == "monitor"
