from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.llm.chat_client import StructuredLLMResponse, TokenUsage
from app.rag_services.hyde_query_transformer import HyDEDocuments, HydeQueryTransformer


class _FakeLLMClient:
    def __init__(self, hypotheses: list[str], *, usage_tokens: int = 42) -> None:
        self._hypotheses = hypotheses
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
        response_model: type[HyDEDocuments],
        model: str | None = None,
        temperature: float = 0.0,
        max_completion_tokens: int = 1_000,
        timeout_seconds: float = 30.0,
        max_attempts: int = 2,
    ) -> StructuredLLMResponse[HyDEDocuments]:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_message": user_message,
                "response_model": response_model,
                "model": model,
                "temperature": temperature,
                "max_completion_tokens": max_completion_tokens,
                "timeout_seconds": timeout_seconds,
                "max_attempts": max_attempts,
            }
        )
        value = response_model(hypothesis=self._hypotheses)
        return StructuredLLMResponse(value=value, usage=TokenUsage(total_tokens=self._usage_tokens))


class _RaisingLLMClient:
    def generate(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError

    def generate_json(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError

    def generate_structured(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("provider unavailable")


def _hypothesis(seed: str = "a") -> str:
    return (seed * 40)[:40]  # >= min_length=32


def _transformer(llm_client: object, num_hypotheses: int = 3) -> HydeQueryTransformer:
    return HydeQueryTransformer(
        llm_client=llm_client,  # type: ignore[arg-type]
        model="gpt-4o-mini",
        prompt_version="bank-policy-v1",
        num_hypotheses=num_hypotheses,
        temperature=0.3,
        max_completion_tokens=600,
        timeout_seconds=12.0,
        max_attempts=2,
    )


def test_transform_requests_the_configured_model_and_budgets() -> None:
    llm_client = _FakeLLMClient([_hypothesis("a"), _hypothesis("b"), _hypothesis("c")])
    transformer = _transformer(llm_client, num_hypotheses=3)

    transformer.transform("what is the wire transfer limit?")

    assert len(llm_client.calls) == 1
    call = llm_client.calls[0]
    assert call["model"] == "gpt-4o-mini"
    assert call["temperature"] == 0.3
    assert call["max_completion_tokens"] == 600
    assert call["timeout_seconds"] == 12.0
    assert call["max_attempts"] == 2
    assert call["response_model"] is HyDEDocuments


def test_transform_returns_exactly_n_distinct_hypotheses() -> None:
    hypotheses = [_hypothesis("a"), _hypothesis("b"), _hypothesis("c")]
    llm_client = _FakeLLMClient(hypotheses)
    transformer = _transformer(llm_client, num_hypotheses=3)

    outcome = transformer.transform("q")

    assert outcome.applied is True
    assert len(outcome.retrieval_texts) == 3
    assert outcome.backend == "hyde:gpt-4o-mini"
    assert outcome.usage_tokens == 42


def test_transform_rejects_blank_query() -> None:
    transformer = _transformer(_FakeLLMClient([_hypothesis()]), num_hypotheses=1)

    with pytest.raises(ValueError):
        transformer.transform("   ")


def test_transform_raises_when_exact_duplicates_reduce_below_the_configured_count() -> None:
    same = _hypothesis("a")
    llm_client = _FakeLLMClient([same, same, _hypothesis("c")])
    transformer = _transformer(llm_client, num_hypotheses=3)

    with pytest.raises(ValueError, match="expected 3"):
        transformer.transform("q")


def test_transform_deduplication_is_case_and_whitespace_insensitive() -> None:
    """Two hypotheses differing only in case/whitespace still collapse to
    one - so a provider that returns near-duplicates gets caught by the
    same exact-count check as literal duplicates, not silently accepted."""
    base = _hypothesis("a")
    llm_client = _FakeLLMClient([base, base.upper(), _hypothesis("b")])
    transformer = _transformer(llm_client, num_hypotheses=3)

    with pytest.raises(ValueError, match="expected 3"):
        transformer.transform("q")


def test_hyde_documents_rejects_too_short_hypotheses() -> None:
    with pytest.raises(ValidationError):
        HyDEDocuments(hypothesis=["too short"])


def test_hyde_documents_rejects_too_long_hypotheses() -> None:
    with pytest.raises(ValidationError):
        HyDEDocuments(hypothesis=["x" * 2_001])


def test_hyde_documents_rejects_more_than_five_hypotheses() -> None:
    with pytest.raises(ValidationError):
        HyDEDocuments(hypothesis=[_hypothesis(str(i)) for i in range(6)])


def test_hyde_documents_rejects_zero_hypotheses() -> None:
    with pytest.raises(ValidationError):
        HyDEDocuments(hypothesis=[])


def test_hyde_documents_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        HyDEDocuments(hypothesis=[_hypothesis()], extra_field="not allowed")  # type: ignore[call-arg]


def test_provider_failure_propagates_from_the_raw_transformer() -> None:
    """HydeQueryTransformer itself does not fail open - that's
    FailOpenQueryTransformer's job (see test_query_transformer.py). A raw
    provider error must propagate so the decorator can catch it."""
    transformer = _transformer(_RaisingLLMClient(), num_hypotheses=1)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        transformer.transform("q")


def test_num_hypotheses_out_of_range_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError):
        _transformer(_FakeLLMClient([]), num_hypotheses=6)
