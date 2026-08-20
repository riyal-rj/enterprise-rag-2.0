"""Shared guardrail types - zero business logic, no I/O.

Every other module in this package (and every call site outside it) depends
only on these types, never on a concrete scanner implementation - that's
what lets ``app.guardrails.llm_guard_adapter`` stay the one place that
imports ``llm_guard``, and lets tests fake ``TextScanner``/``OutputScanner``
without ever constructing a real ML model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Protocol

from app.core.exceptions import AppError

GuardrailMode = Literal["enforce", "monitor"]


class GuardrailAction(StrEnum):
    """What a guardrail decided to do with the text it scanned.

    Deliberately three values, not the blueprint's larger set (no
    ``EXCLUDE``/``ABSTAIN`` as distinct actions) - callers that need
    per-item exclusion (``ContextGuardPipeline.filter_evidence``) or a
    forced abstention (``OutputGuardPipeline`` on ``BLOCK``) implement that
    behavior themselves in response to a plain ``BLOCK``, rather than this
    enum trying to encode every call site's specific fallback.
    """

    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"


class GuardrailStage(StrEnum):
    INPUT = "input"
    CONTEXT = "context"
    OUTPUT = "output"
    TOOL = "tool"
    INGESTION = "ingestion"


class GuardrailCategory(StrEnum):
    PROMPT_INJECTION = "prompt_injection"
    PII = "pii"
    SECRETS = "secrets"
    TOXICITY = "toxicity"
    BANNED_TOPIC = "banned_topic"
    MALICIOUS_URL = "malicious_url"
    PROMPT_LEAK = "prompt_leak"
    UNSAFE_REFUSAL = "unsafe_refusal"
    # Catch-all for the deterministic layer's own checks that don't map
    # cleanly onto one of the categories above (e.g. a decoded-payload
    # smuggling heuristic).
    POLICY = "policy"


@dataclass(frozen=True)
class ScanFinding:
    """One scanner's verdict on one piece of text.

    ``detail`` is a short, human-readable reason code/summary - never the
    matched text itself. Findings feed both the policy decision and (for
    BLOCK/REDACT only) the sanitized security-events audit trail, so this
    type's own discipline is what keeps raw sensitive content out of both.
    """

    category: GuardrailCategory
    score: float
    detector: str
    detail: str | None = None


@dataclass(frozen=True)
class GuardrailDecision:
    """The outcome of running a bag of findings through a
    :class:`app.guardrails.policy.GuardrailPolicy`.

    ``mode`` is the mode that produced ``action`` - in ``"monitor"`` mode,
    ``action`` is always downgraded to ``ALLOW`` (see
    ``GuardrailPolicy.decide``) even when ``findings`` contains a
    block-worthy result, so ``findings``/``would_block`` are what
    ``security_events`` records to make "monitor mode would have blocked
    this" visible without it ever affecting real traffic.
    """

    stage: GuardrailStage
    action: GuardrailAction
    mode: GuardrailMode
    findings: tuple[ScanFinding, ...] = field(default_factory=tuple)
    would_block: bool = False

    @property
    def blocked(self) -> bool:
        return self.action is GuardrailAction.BLOCK


class TextScanner(Protocol):
    """Scans a single piece of text and returns findings. Implemented by
    the deterministic checks (``canonicalizer``/``sensitive_data``), the
    LLM Guard input-side adapter, and hand-written test fakes."""

    @property
    def name(self) -> str: ...

    def scan(self, text: str) -> tuple[ScanFinding, ...]: ...


class OutputScanner(Protocol):
    """Scans generated output against the prompt that produced it - LLM
    Guard's output scanners (``NoRefusal``, ``Sensitive``, ...) need both,
    unlike input scanners."""

    @property
    def name(self) -> str: ...

    def scan(self, *, prompt: str, output: str) -> tuple[ScanFinding, ...]: ...


class GuardrailBlockedError(AppError):
    """Raised when a guardrail's decision is BLOCK and the caller has no
    graceful degradation of its own (input guard: nothing to fall back to
    but rejecting the request; tool guard: nothing to fall back to but
    denying the action).

    Deliberately takes no free-form message - only a fixed, generic public
    string - so ``app.core.exception_handlers``' generic ``str(exc)``
    response body can never leak which category/detector fired or what the
    offending text was. Real diagnostic detail (``stage``/``category``)
    is for server-side logging/``security_events`` only.
    """

    _PUBLIC_MESSAGE = "Your request could not be processed."

    def __init__(self, *, stage: GuardrailStage, category: GuardrailCategory | None = None) -> None:
        super().__init__(self._PUBLIC_MESSAGE)
        self.stage = stage
        self.category = category
