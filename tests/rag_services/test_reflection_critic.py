"""Tests for StructuredReflectionCritic."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.core.llm.chat_client import StructuredLLMResponse, TokenUsage
from app.rag_services.crag import EvidenceChunk, EvidenceOrigin
from app.rag_services.reflection.reflection import SupportLevel
from app.rag_services.reflection.reflection_critic import StructuredReflectionCritic


class _FakeLLMClient:
    def __init__(self, *, payload: dict[str, object] | None = None, usage_tokens: int = 7) -> None:
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


def _critic(llm_client: object, **overrides: object) -> StructuredReflectionCritic:
    defaults: dict[str, object] = dict(
        llm_client=llm_client,
        model="gpt-4o-mini",
        prompt_version="bank-policy-v1",
        timeout_seconds=10.0,
        max_completion_tokens=800,
        max_attempts=2,
        max_evidence_chars=30_000,
    )
    defaults.update(overrides)
    return StructuredReflectionCritic(**defaults)  # type: ignore[arg-type]


def _evidence(text: str = "text", source: str = "a.pdf") -> EvidenceChunk:
    return EvidenceChunk(
        text=text, source=source, page_number=1, retrieval_score=0.9, origin=EvidenceOrigin.POLICY
    )


def _payload(**overrides: object) -> dict[str, object]:
    defaults: dict[str, object] = dict(
        retrieval_needed=False,
        retrieval_query=None,
        evidence_relevance=0.9,
        support_level="full",
        answer_relevance=0.9,
        citation_completeness=0.9,
        utility=5,
        missing_aspects=[],
        unsupported_claims=[],
    )
    defaults.update(overrides)
    return defaults


def test_critique_maps_payload_to_reflection_critique() -> None:
    llm_client = _FakeLLMClient(payload=_payload())
    critic = _critic(llm_client)

    critique = critic.critique("q", (_evidence(),), "answer")

    assert critique.retrieval_needed is False
    assert critique.evidence_relevance == 0.9
    assert critique.support_level is SupportLevel.FULL
    assert critique.usage_tokens == 7


def test_retrieval_query_required_when_retrieval_needed() -> None:
    llm_client = _FakeLLMClient(payload=_payload(retrieval_needed=True, retrieval_query=None))
    critic = _critic(llm_client)

    with pytest.raises(ValidationError):
        critic.critique("q", (_evidence(),), "answer")


def test_retrieval_query_is_cleared_when_not_needed() -> None:
    llm_client = _FakeLLMClient(
        payload=_payload(retrieval_needed=False, retrieval_query="some query")
    )
    critic = _critic(llm_client)

    critique = critic.critique("q", (_evidence(),), "answer")

    assert critique.retrieval_query is None


def test_retrieval_query_present_when_needed() -> None:
    llm_client = _FakeLLMClient(
        payload=_payload(
            retrieval_needed=True,
            retrieval_query="missing KYC threshold policy",
            support_level="partial",
        )
    )
    critic = _critic(llm_client)

    critique = critic.critique("q", (_evidence(),), "answer")

    assert critique.retrieval_needed is True
    assert critique.retrieval_query == "missing KYC threshold policy"


def test_missing_aspects_and_unsupported_claims_are_trimmed_and_deduped() -> None:
    llm_client = _FakeLLMClient(
        payload=_payload(
            missing_aspects=["  extra   whitespace  ", "extra whitespace", ""],
            unsupported_claims=["claim one", "claim one"],
        )
    )
    critic = _critic(llm_client)

    critique = critic.critique("q", (_evidence(),), "answer")

    assert critique.missing_aspects == ("extra whitespace",)
    assert critique.unsupported_claims == ("claim one",)


def test_extra_response_fields_are_rejected() -> None:
    llm_client = _FakeLLMClient(payload=_payload(unexpected="field"))
    critic = _critic(llm_client)

    with pytest.raises(ValidationError):
        critic.critique("q", (_evidence(),), "answer")


def test_utility_out_of_range_is_rejected() -> None:
    llm_client = _FakeLLMClient(payload=_payload(utility=6))
    critic = _critic(llm_client)

    with pytest.raises(ValidationError):
        critic.critique("q", (_evidence(),), "answer")


def test_evidence_is_truncated_to_max_evidence_chars() -> None:
    llm_client = _FakeLLMClient(payload=_payload())
    critic = _critic(llm_client, max_evidence_chars=10)

    critic.critique("q", (_evidence(text="a" * 50),), "answer")

    sent_payload = json.loads(str(llm_client.calls[0]["user_message"]))
    assert len(sent_payload["evidence"][0]["text"]) == 10


def test_timeout_override_is_clamped_to_the_configured_ceiling() -> None:
    """The engine passes the caller's remaining deadline as an override so
    a critic call can't outlive the overall reflection budget - but that
    override must never let a caller extend the timeout past what this
    critic was configured with either."""
    captured: dict[str, object] = {}

    class _CapturingLLMClient:
        def generate(self, *a: object, **k: object) -> None:
            raise NotImplementedError

        def generate_json(self, *a: object, **k: object) -> None:
            raise NotImplementedError

        def generate_structured(self, *args: object, **kwargs: object) -> StructuredLLMResponse:
            captured["timeout_seconds"] = kwargs["timeout_seconds"]
            return StructuredLLMResponse(value=_payload_model(), usage=TokenUsage(total_tokens=1))

    critic = _critic(_CapturingLLMClient(), timeout_seconds=10.0)

    critic.critique("q", (_evidence(),), "answer", timeout_seconds=3.0)
    assert captured["timeout_seconds"] == 3.0  # remaining is smaller - use it

    critic.critique("q", (_evidence(),), "answer", timeout_seconds=100.0)
    assert captured["timeout_seconds"] == 10.0  # remaining is bigger - clamp to configured

    critic.critique("q", (_evidence(),), "answer")
    assert captured["timeout_seconds"] == 10.0  # no override at all - use configured


def _payload_model() -> object:
    from app.rag_services.reflection.reflection_critic import _CritiquePayload

    return _CritiquePayload(**_payload())


def test_question_answer_and_evidence_are_data_not_instructions_in_payload() -> None:
    llm_client = _FakeLLMClient(payload=_payload())
    critic = _critic(llm_client)

    critic.critique("What is the KYC threshold?", (_evidence(text="policy text"),), "The answer.")

    sent_payload = json.loads(str(llm_client.calls[0]["user_message"]))
    assert sent_payload["question"] == "What is the KYC threshold?"
    assert sent_payload["answer"] == "The answer."
    assert sent_payload["evidence"][0]["text"] == "policy text"


def test_cache_namespace_encodes_model_prompt_and_evidence_chars() -> None:
    critic = _critic(_FakeLLMClient(payload=_payload()))

    assert critic.cache_namespace == (
        "critic=gpt-4o-mini:schema=v1:prompt=bank-policy-v1:evidence_chars=30000:"
        "max_tokens=800:timeout=10.0:attempts=2"
    )


def test_cache_namespace_changes_with_every_output_affecting_setting() -> None:
    """Regression: two critics that differ in any setting that can change
    what the critic actually does/returns must never share a cache
    namespace - each override below must produce a namespace distinct from
    the baseline."""
    baseline = _critic(_FakeLLMClient(payload=_payload())).cache_namespace

    assert _critic(_FakeLLMClient(payload=_payload()), model="gpt-4o").cache_namespace != baseline
    assert (
        _critic(_FakeLLMClient(payload=_payload()), prompt_version="bank-policy-v2").cache_namespace
        != baseline
    )
    assert (
        _critic(_FakeLLMClient(payload=_payload()), max_evidence_chars=10_000).cache_namespace
        != baseline
    )
    assert (
        _critic(_FakeLLMClient(payload=_payload()), max_completion_tokens=400).cache_namespace
        != baseline
    )
    assert (
        _critic(_FakeLLMClient(payload=_payload()), timeout_seconds=5.0).cache_namespace != baseline
    )
    assert _critic(_FakeLLMClient(payload=_payload()), max_attempts=1).cache_namespace != baseline
