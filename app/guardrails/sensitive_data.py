"""Deterministic, dependency-free secret/PII detection - the fallback layer
underneath LLM Guard's ``Anonymize``/``Secrets`` scanners (see
``app.guardrails.llm_guard_adapter``). Works standalone with no ML model,
so it's always active regardless of whether this deployment's scanner
models are loaded (``SafetySettings.scanner_models_ready``).

``SensitiveValueVault`` is a request-lifetime-only placeholder store - never
persisted, never logged, never attached to a cache/response/security-event
object. It exists so a legitimate downstream need (e.g. an admin
double-checking "did the user paste an account number") can still see that
*something* was redacted without the raw value ever leaving this process's
memory for this one request.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Callable

from app.guardrails.contracts import GuardrailCategory, ScanFinding

_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# 10-digit Indian mobile number, optionally +91-prefixed.
_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?91[-\s]?)?[6-9]\d{9}(?!\d)")
# PAN: 5 letters, 4 digits, 1 letter (India).
_PAN_PATTERN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
# Aadhaar: 12 digits, optionally space/hyphen-grouped 4-4-4.
_AADHAAR_PATTERN = re.compile(r"(?<!\d)\d{4}[-\s]?\d{4}[-\s]?\d{4}(?!\d)")
# Card-shaped: 13-19 digits, optionally grouped - Luhn-checked below to cut
# false positives against arbitrary long numbers (account/reference ids).
_CARD_CANDIDATE_PATTERN = re.compile(r"(?<!\d)(?:\d[-\s]?){13,19}(?!\d)")
_JWT_PATTERN = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
# Covers both the single-segment form (sk-abc...) and the very common
# two-segment vendor form (sk_live_abc..., pk_test_abc...). The optional
# middle segment is why this isn't just one separator: without it, every
# Stripe-style key would slip through. The trailing 16+ alphanumeric run is
# what keeps ordinary prose ("key_value_pairs") from matching.
_GENERIC_API_KEY_PATTERN = re.compile(
    r"\b(?:sk|pk|api|key)[-_](?:[A-Za-z0-9]+[-_])?[A-Za-z0-9]{16,}\b", re.IGNORECASE
)


def _luhn_valid(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        digit = int(char)
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


class SensitiveValueVault:
    """Request-scoped placeholder <-> raw-value mapping. Construct one per
    request, use it for the duration of that request only, discard it -
    never share an instance across requests or persist its contents."""

    __slots__ = ("_values",)

    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def store(self, kind: str, raw_value: str) -> str:
        placeholder = f"[REDACTED:{kind.upper()}:{secrets.token_hex(4)}]"
        self._values[placeholder] = raw_value
        return placeholder

    def resolve(self, placeholder: str) -> str | None:
        return self._values.get(placeholder)

    def __len__(self) -> int:
        return len(self._values)


class RegexSensitiveDataScanner:
    """Fast, dependency-free detector for structured secrets/PII. Findings
    only - does not itself redact ``text`` (unlike LLM Guard's ``Anonymize``,
    which returns sanitized text); callers that want redaction use
    :class:`SensitiveValueVault` directly against the same patterns via
    :func:`tokenize`.
    """

    @property
    def name(self) -> str:
        return "deterministic.regex_sensitive_data"

    def scan(self, text: str) -> tuple[ScanFinding, ...]:
        findings: list[ScanFinding] = []
        if _JWT_PATTERN.search(text) or _GENERIC_API_KEY_PATTERN.search(text):
            findings.append(
                ScanFinding(
                    category=GuardrailCategory.SECRETS,
                    score=1.0,
                    detector=self.name,
                    detail="credential_shaped_token",
                )
            )
        if _EMAIL_PATTERN.search(text) or _PHONE_PATTERN.search(text):
            findings.append(
                ScanFinding(
                    category=GuardrailCategory.PII,
                    score=0.8,
                    detector=self.name,
                    detail="contact_identifier",
                )
            )
        if _PAN_PATTERN.search(text) or _AADHAAR_PATTERN.search(text):
            findings.append(
                ScanFinding(
                    category=GuardrailCategory.PII,
                    score=0.95,
                    detector=self.name,
                    detail="government_id",
                )
            )
        for match in _CARD_CANDIDATE_PATTERN.finditer(text):
            digits = re.sub(r"[-\s]", "", match.group(0))
            if 13 <= len(digits) <= 19 and _luhn_valid(digits):
                findings.append(
                    ScanFinding(
                        category=GuardrailCategory.PII,
                        score=0.9,
                        detector=self.name,
                        detail="card_number",
                    )
                )
                break
        return tuple(findings)


def _store_match(vault: SensitiveValueVault, kind: str) -> Callable[[re.Match[str]], str]:
    def _replace(match: re.Match[str]) -> str:
        return vault.store(kind, match.group(0))

    return _replace


def tokenize(text: str, vault: SensitiveValueVault) -> str:
    """Replace every pattern :class:`RegexSensitiveDataScanner` recognizes
    with a stable placeholder, recording the raw value in ``vault``. Used
    by the input pipeline's REDACT path - the LLM never sees the raw
    secret/PII value, only the placeholder."""
    result = text
    for kind, pattern in (
        ("jwt", _JWT_PATTERN),
        ("api_key", _GENERIC_API_KEY_PATTERN),
        ("email", _EMAIL_PATTERN),
        ("phone", _PHONE_PATTERN),
        ("pan", _PAN_PATTERN),
        ("aadhaar", _AADHAAR_PATTERN),
    ):
        result = pattern.sub(_store_match(vault, kind), result)

    def _replace_card(match: re.Match[str]) -> str:
        digits = re.sub(r"[-\s]", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_valid(digits):
            return vault.store("card", match.group(0))
        return match.group(0)

    result = _CARD_CANDIDATE_PATTERN.sub(_replace_card, result)
    return result
