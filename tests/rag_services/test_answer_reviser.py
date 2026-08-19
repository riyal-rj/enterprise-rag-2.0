"""Tests for StructuredGroundedAnswerReviser."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.core.llm.chat_client import StructuredLLMResponse, TokenUsage
from app.rag_services.crag import EvidenceChunk, EvidenceOrigin
from app.rag_services.reflection.answer_reviser import StructuredGroundedAnswerReviser
from app.rag_services.reflection.reflection import ReflectionCritique, SupportLevel


class _FakeLLMClient:
    def __init__(self, *, payload: dict[str, object] | None = None, usage_tokens: int = 11) -> None:
        self._payload = payload
        self._usage_tokens = usage_tokens
        self.calls: list[dict[str, object]] = []

    def generate(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError

    def generate_json(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError

    def generate_structured(
        self,
        system_prompt: str,
        user_message: str,
        *,
        response_model: type,
        model: str | None = None,
        temperature: float = 0.0,
        max_completion_tokens: int = 1_000,
        timeout_seconds: float = 30.0,
        max_attempts: int = 2,
    ) -> StructuredLLMResponse:
        self.calls.append({"system_prompt": system_prompt, "user_message": user_message})
        assert self._payload is not None
        value = response_model(**self._payload)
        return StructuredLLMResponse(value=value, usage=TokenUsage(total_tokens=self._usage_tokens))


def _reviser(llm_client: object, **overrides: object) -> StructuredGroundedAnswerReviser:
    defaults: dict[str, object] = dict(
        llm_client=llm_client,
        model="gpt-4o",
        prompt_version="bank-policy-v1",
        timeout_seconds=10.0,
        max_completion_tokens=1_500,
        max_attempts=2,
        max_evidence_chars=30_000,
    )
    defaults.update(overrides)
    return StructuredGroundedAnswerReviser(**defaults)  # type: ignore[arg-type]


def _evidence(text: str = "text", source: str = "a.pdf") -> EvidenceChunk:
    return EvidenceChunk(
        text=text, source=source, page_number=1, retrieval_score=0.9, origin=EvidenceOrigin.POLICY
    )


def _critique(**overrides: object) -> ReflectionCritique:
    defaults: dict[str, object] = dict(
        retrieval_needed=False,
        retrieval_query=None,
        evidence_relevance=0.9,
        support_level=SupportLevel.PARTIAL,
        answer_relevance=0.6,
        citation_completeness=0.5,
        utility=3,
        missing_aspects=("deadline",),
        unsupported_claims=("a claim",),
    )
    defaults.update(overrides)
    return ReflectionCritique(**defaults)  # type: ignore[arg-type]


def test_revise_returns_revised_answer_and_token_count() -> None:
    llm_client = _FakeLLMClient(payload={"answer": "Revised, fully cited answer."})
    reviser = _reviser(llm_client)

    answer, tokens = reviser.revise("q", (_evidence(),), "old answer", _critique())

    assert answer == "Revised, fully cited answer."
    assert tokens == 11


def test_answer_is_stripped_of_surrounding_whitespace() -> None:
    llm_client = _FakeLLMClient(payload={"answer": "  padded answer  "})
    reviser = _reviser(llm_client)

    answer, _ = reviser.revise("q", (_evidence(),), "old answer", _critique())

    assert answer == "padded answer"


def test_empty_answer_is_rejected() -> None:
    llm_client = _FakeLLMClient(payload={"answer": ""})
    reviser = _reviser(llm_client)

    with pytest.raises(ValidationError):
        reviser.revise("q", (_evidence(),), "old answer", _critique())


def test_extra_response_fields_are_rejected() -> None:
    llm_client = _FakeLLMClient(payload={"answer": "ok", "unexpected": "field"})
    reviser = _reviser(llm_client)

    with pytest.raises(ValidationError):
        reviser.revise("q", (_evidence(),), "old answer", _critique())


def test_payload_carries_evidence_previous_answer_and_feedback() -> None:
    llm_client = _FakeLLMClient(payload={"answer": "revised"})
    reviser = _reviser(llm_client)

    reviser.revise(
        "What is the KYC threshold?",
        (_evidence(text="policy text", source="policy.pdf"),),
        "The previous answer.",
        _critique(missing_aspects=("filing deadline",), unsupported_claims=("bad claim",)),
    )

    sent_payload = json.loads(str(llm_client.calls[0]["user_message"]))
    assert sent_payload["question"] == "What is the KYC threshold?"
    assert sent_payload["previous_answer"] == "The previous answer."
    assert sent_payload["evidence"][0]["text"] == "policy text"
    assert sent_payload["evidence"][0]["source"] == "policy.pdf"
    assert sent_payload["feedback"]["missing_aspects"] == ["filing deadline"]
    assert sent_payload["feedback"]["unsupported_claims"] == ["bad claim"]
    assert sent_payload["feedback"]["support_level"] == "partial"


def test_cache_namespace_encodes_model_and_prompt_version() -> None:
    reviser = _reviser(_FakeLLMClient(payload={"answer": "ok"}))

    assert reviser.cache_namespace == (
        "reviser=gpt-4o:schema=v1:prompt=bank-policy-v1:evidence_chars=30000:"
        "max_tokens=1500:timeout=10.0:attempts=2"
    )


def test_cache_namespace_changes_with_every_output_affecting_setting() -> None:
    baseline = _reviser(_FakeLLMClient(payload={"answer": "ok"})).cache_namespace

    assert (
        _reviser(_FakeLLMClient(payload={"answer": "ok"}), model="gpt-4o-mini").cache_namespace
        != baseline
    )
    assert (
        _reviser(
            _FakeLLMClient(payload={"answer": "ok"}), prompt_version="bank-policy-v2"
        ).cache_namespace
        != baseline
    )
    assert (
        _reviser(
            _FakeLLMClient(payload={"answer": "ok"}), max_evidence_chars=10_000
        ).cache_namespace
        != baseline
    )
    assert (
        _reviser(
            _FakeLLMClient(payload={"answer": "ok"}), max_completion_tokens=500
        ).cache_namespace
        != baseline
    )
    assert (
        _reviser(_FakeLLMClient(payload={"answer": "ok"}), timeout_seconds=5.0).cache_namespace
        != baseline
    )
    assert (
        _reviser(_FakeLLMClient(payload={"answer": "ok"}), max_attempts=1).cache_namespace
        != baseline
    )


def test_evidence_is_truncated_to_max_evidence_chars() -> None:
    llm_client = _FakeLLMClient(payload={"answer": "ok"})
    reviser = _reviser(llm_client, max_evidence_chars=10)

    reviser.revise("q", (_evidence(text="a" * 50),), "old answer", _critique())

    sent_payload = json.loads(str(llm_client.calls[0]["user_message"]))
    assert len(sent_payload["evidence"][0]["text"]) == 10


def test_timeout_override_is_clamped_to_the_configured_ceiling() -> None:
    captured: dict[str, object] = {}

    class _CapturingLLMClient:
        def generate(self, *a: object, **k: object) -> None:
            raise NotImplementedError

        def generate_json(self, *a: object, **k: object) -> None:
            raise NotImplementedError

        def generate_structured(self, *args: object, **kwargs: object) -> StructuredLLMResponse:
            captured["timeout_seconds"] = kwargs["timeout_seconds"]
            from app.rag_services.reflection.answer_reviser import _RevisedAnswerPayload

            return StructuredLLMResponse(
                value=_RevisedAnswerPayload(answer="ok"), usage=TokenUsage(total_tokens=1)
            )

    reviser = _reviser(_CapturingLLMClient(), timeout_seconds=10.0)

    reviser.revise("q", (_evidence(),), "old", _critique(), timeout_seconds=3.0)
    assert captured["timeout_seconds"] == 3.0

    reviser.revise("q", (_evidence(),), "old", _critique(), timeout_seconds=100.0)
    assert captured["timeout_seconds"] == 10.0
