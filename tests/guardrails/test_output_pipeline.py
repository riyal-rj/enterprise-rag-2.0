"""Output pipeline: the last gate before an answer can be cached or served."""

from __future__ import annotations

import pytest

from app.guardrails.contracts import (
    GuardrailBlockedError,
    GuardrailCategory,
    GuardrailStage,
    ScanFinding,
)
from app.guardrails.output_pipeline import NoOpOutputGuardPipeline, OutputGuardPipeline
from app.guardrails.policy import GuardrailPolicy
from app.guardrails.sensitive_data import RegexSensitiveDataScanner
from tests.guardrails.fakes import FakeOutputScanner, FakeTextScanner


def _pipeline(
    *,
    deterministic: tuple[object, ...] = (),
    ml_scanner: object | None = None,
) -> OutputGuardPipeline:
    return OutputGuardPipeline(
        deterministic_scanners=deterministic,  # type: ignore[arg-type]
        ml_scanner=ml_scanner,  # type: ignore[arg-type]
        policy=GuardrailPolicy(),
    )


def test_clean_answer_passes_through_unchanged() -> None:
    answer = "Leave encashment is capped at 15 days [a.pdf]."

    assert _pipeline().apply(prompt="q", answer=answer, mode="enforce") == answer


def test_output_scanner_receives_both_prompt_and_answer() -> None:
    """LLM Guard's output scanners (NoRefusal, Sensitive) need the prompt
    for context, unlike input scanners."""
    scanner = FakeOutputScanner()
    _pipeline(ml_scanner=scanner).apply(prompt="the question", answer="the answer", mode="enforce")

    assert scanner.calls == [("the question", "the answer")]


def test_blocking_finding_raises_guardrail_blocked_error() -> None:
    scanner = FakeOutputScanner((ScanFinding(GuardrailCategory.PROMPT_LEAK, 0.9, "fake"),))

    with pytest.raises(GuardrailBlockedError) as excinfo:
        _pipeline(ml_scanner=scanner).apply(
            prompt="q", answer="here is my system prompt", mode="enforce"
        )

    assert excinfo.value.stage is GuardrailStage.OUTPUT
    assert excinfo.value.category is GuardrailCategory.PROMPT_LEAK


def test_monitor_mode_never_raises() -> None:
    scanner = FakeOutputScanner((ScanFinding(GuardrailCategory.PROMPT_LEAK, 0.9, "fake"),))

    result = _pipeline(ml_scanner=scanner).apply(prompt="q", answer="leaky", mode="monitor")

    assert result == "leaky"


def test_pii_in_the_answer_is_redacted_not_blocked() -> None:
    """Catches PII the model echoed or hallucinated that wasn't in the
    retrieved evidence."""
    pipeline = _pipeline(deterministic=(RegexSensitiveDataScanner(),))

    result = pipeline.apply(
        prompt="q", answer="Contact the customer at foo@bank.example", mode="enforce"
    )

    assert "foo@bank.example" not in result
    assert "[REDACTED:EMAIL:" in result


def test_deterministic_scanners_see_the_answer_text() -> None:
    scanner = FakeTextScanner()
    _pipeline(deterministic=(scanner,)).apply(prompt="q", answer="the answer", mode="enforce")

    assert scanner.calls == ["the answer"]


def test_unsafe_refusal_finding_does_not_block_a_legitimate_abstention() -> None:
    """A grounded "the evidence is insufficient" answer is exactly what
    this system is supposed to produce when evidence is thin - NoRefusal
    flagging it must never turn it into a block."""
    scanner = FakeOutputScanner(
        (ScanFinding(GuardrailCategory.UNSAFE_REFUSAL, 0.95, "llm_guard.NoRefusal"),)
    )
    answer = "The supplied approved evidence is insufficient to determine this point."

    assert _pipeline(ml_scanner=scanner).apply(
        prompt="q", answer=answer, mode="enforce"
    ) == answer


def test_noop_pipeline_returns_the_answer_untouched() -> None:
    assert NoOpOutputGuardPipeline().apply(prompt="q", answer="a") == "a"
