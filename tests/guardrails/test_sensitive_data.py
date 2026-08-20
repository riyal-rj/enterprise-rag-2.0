"""Deterministic secret/PII layer - no ML model, no fakes needed."""

from __future__ import annotations

from app.guardrails.contracts import GuardrailCategory
from app.guardrails.sensitive_data import (
    RegexSensitiveDataScanner,
    SensitiveValueVault,
    tokenize,
)


def _categories(text: str) -> set[GuardrailCategory]:
    return {finding.category for finding in RegexSensitiveDataScanner().scan(text)}


def test_detects_jwt_as_a_secret() -> None:
    token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abc-DEF_123"

    assert GuardrailCategory.SECRETS in _categories(f"my token is {token}")


def test_detects_api_key_shaped_string_as_a_secret() -> None:
    assert GuardrailCategory.SECRETS in _categories("use sk-abcdefghijklmnopqrstuvwxyz123")


def test_detects_two_segment_vendor_key_as_a_secret() -> None:
    """Stripe-style sk_live_.../pk_test_... is the most common real-world
    key shape - a single-separator pattern would miss all of them."""
    assert GuardrailCategory.SECRETS in _categories("use sk_live_ABCDEFGHIJKLMNOPQRST now")


def test_ordinary_snake_case_identifier_is_not_a_false_positive_key() -> None:
    assert GuardrailCategory.SECRETS not in _categories("iterate over the key_value_pairs list")


def test_detects_email_and_phone_as_pii() -> None:
    assert GuardrailCategory.PII in _categories("reach me at foo.bar@example.com")
    assert GuardrailCategory.PII in _categories("call 9876543210 tomorrow")


def test_detects_pan_and_aadhaar_as_pii() -> None:
    assert GuardrailCategory.PII in _categories("my PAN is ABCDE1234F")
    assert GuardrailCategory.PII in _categories("aadhaar 1234 5678 9012")


def test_luhn_valid_card_number_is_flagged() -> None:
    assert GuardrailCategory.PII in _categories("card 4111 1111 1111 1111 expires soon")


def test_luhn_invalid_long_number_is_not_flagged_as_a_card() -> None:
    """A 16-digit reference/order id that fails the Luhn check must not be
    reported as a card number - the whole point of Luhn-checking rather
    than matching any long digit run."""
    findings = RegexSensitiveDataScanner().scan("reference 1234567812345678 was processed")

    assert not any(f.detail == "card_number" for f in findings)


def test_ordinary_banking_question_produces_no_findings() -> None:
    assert RegexSensitiveDataScanner().scan("what is the leave encashment policy?") == ()


def test_findings_never_contain_the_matched_raw_text() -> None:
    """The detail field feeds the sanitized security-events audit trail -
    it must carry a reason code, never the secret itself."""
    findings = RegexSensitiveDataScanner().scan("my email is secret.person@bank.example")

    assert findings
    for finding in findings:
        assert "secret.person@bank.example" not in (finding.detail or "")
        assert "secret.person" not in repr(finding)


def test_tokenize_replaces_values_with_placeholders_and_records_them() -> None:
    vault = SensitiveValueVault()

    result = tokenize("email foo@bar.com and card 4111111111111111", vault)

    assert "foo@bar.com" not in result
    assert "4111111111111111" not in result
    assert "[REDACTED:EMAIL:" in result
    assert "[REDACTED:CARD:" in result
    assert len(vault) == 2


def test_vault_resolves_a_placeholder_back_to_its_raw_value() -> None:
    vault = SensitiveValueVault()
    placeholder = vault.store("email", "foo@bar.com")

    assert vault.resolve(placeholder) == "foo@bar.com"
    assert vault.resolve("[REDACTED:EMAIL:doesnotexist]") is None


def test_each_placeholder_is_unique_even_for_the_same_value() -> None:
    vault = SensitiveValueVault()

    first = vault.store("email", "foo@bar.com")
    second = vault.store("email", "foo@bar.com")

    assert first != second


def test_tokenize_leaves_a_luhn_invalid_number_alone() -> None:
    vault = SensitiveValueVault()

    result = tokenize("reference 1234567812345678 was processed", vault)

    assert "1234567812345678" in result
