"""The only module in this codebase that imports ``llm_guard``.

Every other guardrail module depends on ``TextScanner``/``OutputScanner``
from ``app.guardrails.contracts``, never on ``llm_guard`` itself - that
isolation is what lets the rest of the test suite fake these Protocols
without ever triggering a real transformer-model download (see
``tests/guardrails/fakes.py`` and ``tests/guardrails/test_llm_guard_adapter.py``,
the one test file that imports this module).

Constructing ``LLMGuardInputScanner``/``LLMGuardOutputScanner`` triggers
model loads (``PromptInjection()`` etc. download/load transformer weights on
first construction) - callers MUST only build these once, at process start,
via ``app.api.deps``'s ``@lru_cache`` factories, never per-request and never
at import time. This module itself does no module-level instantiation.
"""

from __future__ import annotations

import logging

from llm_guard.input_scanners import Anonymize as InputAnonymize
from llm_guard.input_scanners import BanTopics, PromptInjection, Secrets
from llm_guard.input_scanners import Toxicity as InputToxicity
from llm_guard.model import Model
from llm_guard.output_scanners import Deanonymize, MaliciousURLs, NoRefusal, Sensitive
from llm_guard.output_scanners import Toxicity as OutputToxicity
from llm_guard.vault import Vault

from app.core.config.safety import SafetySettings
from app.guardrails.contracts import GuardrailCategory, ScanFinding

logger = logging.getLogger(__name__)

# MaliciousURLs' built-in default model (DunnBC22/codebert-base-Malicious_URLs)
# 401s on the Hub as of this writing - that author's repo is gone. This is
# the same weights re-hosted by ProtectAI in ONNX format (confirmed
# reachable) - both the tokenizer and the classifier need to come from here,
# since llm-guard's non-ONNX tokenizer load always uses `path` regardless of
# `use_onnx`. Pinned to a specific revision (not "main"), matching every
# other model pin in this module.
# llm-guard's bundled default (protectai/deberta-v3-base-prompt-injection-v2)
# has a severe false-positive problem on ordinary first-person "my <X> is
# <value>, <question>" banking self-disclosure sentences ("My card number is
# 4111 1111 1111 1111, what is the dispute window for it?" scores ~1.0 -
# indistinguishable from a genuine injection, also ~1.0, so no threshold
# separates them). Benchmarked against this deployment's adversarial +
# false-positive test cases:
#   - v2 (current default): flagged 5/6 self-disclosure false positives.
#   - deepset/deberta-v3-base-injection: 1/6 self-disclosure FPs, 1/7 on
#     ordinary benign banking questions (flagged an email-sharing request).
#   - protectai/deberta-v3-base-prompt-injection (v1, this one): 1/6
#     self-disclosure FPs (a bare, contextless "card 4111 1111 1111 1111"
#     fragment with no verb/question - a defensible flag), 0/7 on ordinary
#     benign banking questions. Same 5/5 recall on genuine injections as v2.
# See tests/guardrails/test_llm_guard_adapter.py's
# test_real_input_scanners_do_not_flag_a_benign_self_disclosure_question.
_PROMPT_INJECTION_MODEL = Model(
    path="protectai/deberta-v3-base-prompt-injection",
    revision="373b6af0f8d16739cff5de28be326652246bfaa3",
    onnx_path="protectai/deberta-v3-base-prompt-injection",
    onnx_revision="373b6af0f8d16739cff5de28be326652246bfaa3",
    onnx_subfolder="onnx",
    onnx_filename="model.onnx",
    pipeline_kwargs={
        "return_token_type_ids": False,
        "max_length": 512,
        "truncation": True,
    },
)

_MALICIOUS_URLS_ONNX_MODEL = Model(
    path="protectai/codebert-base-Malicious_URLs-onnx",
    revision="6e77d4e5b602736f8ec048463242e41894884844",
    onnx_path="protectai/codebert-base-Malicious_URLs-onnx",
    onnx_revision="6e77d4e5b602736f8ec048463242e41894884844",
    pipeline_kwargs={
        "top_k": None,
        "return_token_type_ids": False,
        "max_length": 128,
        "truncation": True,
    },
)

# Maps each underlying llm-guard scanner's class name (the key scan_prompt/
# scan_output's results dicts use) to the GuardrailCategory it represents -
# every input/output scanner wired below must have an entry here.
_INPUT_SCANNER_CATEGORY: dict[str, GuardrailCategory] = {
    "PromptInjection": GuardrailCategory.PROMPT_INJECTION,
    "Anonymize": GuardrailCategory.PII,
    "Secrets": GuardrailCategory.SECRETS,
    "Toxicity": GuardrailCategory.TOXICITY,
    "BanTopics": GuardrailCategory.BANNED_TOPIC,
}
_OUTPUT_SCANNER_CATEGORY: dict[str, GuardrailCategory] = {
    "Deanonymize": GuardrailCategory.PII,
    "NoRefusal": GuardrailCategory.UNSAFE_REFUSAL,
    "Sensitive": GuardrailCategory.PII,
    "MaliciousURLs": GuardrailCategory.MALICIOUS_URL,
    "Toxicity": GuardrailCategory.TOXICITY,
}


class LLMGuardInputScanner:
    """Wraps a fixed set of llm-guard input scanners behind
    :class:`app.guardrails.contracts.TextScanner`."""

    def __init__(self, scanners: list[object]) -> None:
        self._scanners = scanners

    @property
    def name(self) -> str:
        return "llm_guard.input"

    def scan(self, text: str) -> tuple[ScanFinding, ...]:
        findings: list[ScanFinding] = []
        for scanner in self._scanners:
            scanner_name = type(scanner).__name__
            try:
                _sanitized, is_valid, score = scanner.scan(text)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 - one scanner's failure must not sink the rest
                logger.warning(
                    "guardrails.llm_guard_scanner_error",
                    extra={"scanner": scanner_name, "error_type": type(exc).__name__},
                )
                continue
            if not is_valid:
                findings.append(
                    ScanFinding(
                        category=_INPUT_SCANNER_CATEGORY.get(
                            scanner_name, GuardrailCategory.POLICY
                        ),
                        score=float(score),
                        detector=f"llm_guard.{scanner_name}",
                    )
                )
        return tuple(findings)


class LLMGuardOutputScanner:
    """Wraps a fixed set of llm-guard output scanners behind
    :class:`app.guardrails.contracts.OutputScanner`."""

    def __init__(self, scanners: list[object]) -> None:
        self._scanners = scanners

    @property
    def name(self) -> str:
        return "llm_guard.output"

    def scan(self, *, prompt: str, output: str) -> tuple[ScanFinding, ...]:
        findings: list[ScanFinding] = []
        for scanner in self._scanners:
            scanner_name = type(scanner).__name__
            try:
                _sanitized, is_valid, score = scanner.scan(prompt, output)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 - one scanner's failure must not sink the rest
                logger.warning(
                    "guardrails.llm_guard_scanner_error",
                    extra={"scanner": scanner_name, "error_type": type(exc).__name__},
                )
                continue
            if not is_valid:
                category = _OUTPUT_SCANNER_CATEGORY.get(scanner_name, GuardrailCategory.POLICY)
                # NoRefusal is monitoring-only: a grounded abstention
                # ("insufficient evidence") is a legitimate, expected
                # banking-RAG answer, not a defect - GuardrailPolicy never
                # treats UNSAFE_REFUSAL as block/redact-worthy (see
                # app.guardrails.policy), so this finding is recorded for
                # security_events visibility only.
                findings.append(
                    ScanFinding(
                        category=category,
                        score=float(score),
                        detector=f"llm_guard.{scanner_name}",
                    )
                )
        return tuple(findings)


def build_llm_guard_input_scanner(settings: SafetySettings) -> LLMGuardInputScanner:
    """Triggers real model loads - call exactly once, from a process-wide
    ``@lru_cache`` factory (see ``app.api.deps.get_llm_guard_input_scanner``),
    never per-request."""
    scanners: list[object] = [
        # See _PROMPT_INJECTION_MODEL's docstring for why this isn't
        # llm-guard's bundled default model.
        PromptInjection(
            model=_PROMPT_INJECTION_MODEL,
            threshold=settings.prompt_injection_threshold,
            use_onnx=True,
        ),
        InputAnonymize(Vault()),
        Secrets(),
        InputToxicity(threshold=settings.toxicity_threshold),
    ]
    if settings.banned_topics:
        scanners.append(BanTopics(list(settings.banned_topics)))
    return LLMGuardInputScanner(scanners)


def build_llm_guard_output_scanner(
    settings: SafetySettings, *, vault: Vault | None = None
) -> LLMGuardOutputScanner:
    """Triggers real model loads - call exactly once, from a process-wide
    ``@lru_cache`` factory (see
    ``app.api.deps.get_llm_guard_output_scanner``), never per-request.
    ``vault`` should be the same ``Vault`` instance passed to
    :class:`llm_guard.input_scanners.Anonymize` if a caller wants
    ``Deanonymize`` to have anything to reverse - defaults to a fresh
    (permanently empty, in this deployment's default posture of never
    deanonymizing) vault otherwise.
    """
    scanners: list[object] = [
        Deanonymize(vault or Vault()),
        NoRefusal(),
        Sensitive(),
        # See _MALICIOUS_URLS_ONNX_MODEL's docstring: the scanner's built-in
        # default model repo is gone from the Hub, so both the tokenizer and
        # classifier are pointed at the ProtectAI ONNX mirror instead.
        # Requires the llm-guard[onnxruntime] extra (see pyproject.toml).
        MaliciousURLs(model=_MALICIOUS_URLS_ONNX_MODEL, use_onnx=True),
        OutputToxicity(threshold=settings.output_toxicity_threshold),
    ]
    return LLMGuardOutputScanner(scanners)
