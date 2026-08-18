"""Tests for StructuredLLMRetrievalGrader."""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from app.core.llm.chat_client import StructuredLLMResponse, TokenUsage
from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.crag.crag import CRAGDecision, GraderParseError
from app.rag_services.crag.crag_retrieval_grader import StructuredLLMRetrievalGrader


class _FakeLLMClient:
    def __init__(self, *, payload: dict[str, object] | None = None, usage_tokens: int = 5) -> None:
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


class _RaisingLLMClient:
    def generate(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError

    def generate_json(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError

    def generate_structured(self, *args: object, **kwargs: object) -> None:
        raise TimeoutError("grader call timed out")


class _FlakyLLMClient:
    """Returns a malformed payload on its first call, a valid one on every
    call after - lets tests prove the grader's single retry actually
    recovers instead of merely re-raising the same failure."""

    def __init__(self, *, bad_payload: dict[str, object], good_payload: dict[str, object]) -> None:
        self._bad_payload = bad_payload
        self._good_payload = good_payload
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
        payload = self._bad_payload if len(self.calls) == 1 else self._good_payload
        value = response_model(**payload)
        return StructuredLLMResponse(value=value, usage=TokenUsage(total_tokens=5))


def _grader(llm_client: object, **overrides: object) -> StructuredLLMRetrievalGrader:
    defaults: dict[str, object] = dict(
        llm_client=llm_client,
        model="gpt-4o-mini",
        correct_threshold=0.7,
        ambiguous_threshold=0.5,
        max_chunks=8,
        timeout_seconds=10.0,
        max_completion_tokens=800,
        max_attempts=2,
        prompt_version="bank-policy-v1",
    )
    defaults.update(overrides)
    return StructuredLLMRetrievalGrader(**defaults)  # type: ignore[arg-type]


def _chunk(text: str = "text", source: str = "a.pdf") -> RetrievedChunk:
    return RetrievedChunk(text=text, source=source, score=0.9, page_number=1)


def _chunk_payload(
    index: int, *, relevance: float, supports_question: bool, reason_code: str = "directly_relevant"
) -> dict[str, object]:
    return {
        "index": index,
        "relevance": relevance,
        "supports_question": supports_question,
        "reason_code": reason_code,
    }


def test_no_chunks_returns_incorrect_without_calling_the_llm() -> None:
    llm_client = _FakeLLMClient(payload={"coverage": 1.0, "chunks": []})
    grader = _grader(llm_client)

    grade = grader.grade("question", [])

    assert grade.decision is CRAGDecision.INCORRECT
    assert grade.coverage == 0.0
    assert llm_client.calls == []


def test_high_coverage_with_support_maps_to_correct() -> None:
    llm_client = _FakeLLMClient(
        payload={
            "coverage": 0.9,
            "chunks": [_chunk_payload(0, relevance=0.9, supports_question=True)],
        }
    )
    grader = _grader(llm_client)

    grade = grader.grade("q", [_chunk()])

    assert grade.decision is CRAGDecision.CORRECT


def test_low_coverage_without_support_maps_to_incorrect() -> None:
    llm_client = _FakeLLMClient(
        payload={
            "coverage": 0.1,
            "chunks": [
                _chunk_payload(0, relevance=0.1, supports_question=False, reason_code="irrelevant")
            ],
        }
    )
    grader = _grader(llm_client)

    grade = grader.grade("q", [_chunk()])

    assert grade.decision is CRAGDecision.INCORRECT


def test_middle_coverage_maps_to_ambiguous() -> None:
    llm_client = _FakeLLMClient(
        payload={
            "coverage": 0.6,
            "chunks": [_chunk_payload(0, relevance=0.6, supports_question=True)],
        }
    )
    grader = _grader(llm_client)

    grade = grader.grade("q", [_chunk()])

    assert grade.decision is CRAGDecision.AMBIGUOUS


def test_missing_index_is_rejected() -> None:
    """Two chunks selected, grader only graded index 0 - the completeness
    check must catch this even though nothing about a single valid index is
    individually invalid. The grader deterministically returns the same bad
    payload every call, so both the initial attempt and its one retry fail
    identically, and the failure surfaces as GraderParseError (not a bare
    ValueError) once the retry budget is exhausted."""
    llm_client = _FakeLLMClient(
        payload={
            "coverage": 0.9,
            "chunks": [_chunk_payload(0, relevance=0.9, supports_question=True)],
        }
    )
    grader = _grader(llm_client)

    with pytest.raises(GraderParseError, match="unparseable/invalid output"):
        grader.grade("q", [_chunk("a"), _chunk("b")])
    assert len(llm_client.calls) == 2  # initial attempt + one retry


def test_out_of_range_index_is_rejected() -> None:
    llm_client = _FakeLLMClient(
        payload={
            "coverage": 0.9,
            "chunks": [_chunk_payload(5, relevance=0.9, supports_question=True)],
        }
    )
    grader = _grader(llm_client)

    with pytest.raises(GraderParseError, match="unparseable/invalid output"):
        grader.grade("q", [_chunk()])
    assert len(llm_client.calls) == 2


def test_duplicate_indices_are_rejected() -> None:
    llm_client = _FakeLLMClient(
        payload={
            "coverage": 0.9,
            "chunks": [
                _chunk_payload(0, relevance=0.9, supports_question=True),
                _chunk_payload(0, relevance=0.1, supports_question=False),
            ],
        }
    )
    grader = _grader(llm_client)

    with pytest.raises(GraderParseError) as exc_info:
        grader.grade("q", [_chunk()])
    assert isinstance(exc_info.value.__cause__, ValidationError)
    assert len(llm_client.calls) == 2


def test_extra_response_fields_are_rejected() -> None:
    llm_client = _FakeLLMClient(
        payload={
            "coverage": 0.9,
            "chunks": [_chunk_payload(0, relevance=0.9, supports_question=True)],
            "unexpected": "field",
        }
    )
    grader = _grader(llm_client)

    with pytest.raises(GraderParseError) as exc_info:
        grader.grade("q", [_chunk()])
    assert isinstance(exc_info.value.__cause__, ValidationError)


def test_chunk_payload_extra_fields_are_rejected() -> None:
    llm_client = _FakeLLMClient(
        payload={
            "coverage": 0.9,
            "chunks": [
                {
                    "index": 0,
                    "relevance": 0.9,
                    "supports_question": True,
                    "reason_code": "directly_relevant",
                    "unexpected": "field",
                }
            ],
        }
    )
    grader = _grader(llm_client)

    with pytest.raises(GraderParseError) as exc_info:
        grader.grade("q", [_chunk()])
    assert isinstance(exc_info.value.__cause__, ValidationError)


def test_malformed_output_recovers_if_the_retry_succeeds() -> None:
    """The one-retry budget exists to smooth over a single flaky/malformed
    generation, not just to fail slower - a bad first attempt followed by a
    valid second must return a real grade, not exhaust into
    GraderParseError."""
    llm_client = _FlakyLLMClient(
        bad_payload={
            "coverage": 0.9,
            "chunks": [_chunk_payload(5, relevance=0.9, supports_question=True)],
        },
        good_payload={
            "coverage": 0.9,
            "chunks": [_chunk_payload(0, relevance=0.9, supports_question=True)],
        },
    )
    grader = _grader(llm_client)

    grade = grader.grade("q", [_chunk()])

    assert grade.decision is CRAGDecision.CORRECT
    assert len(llm_client.calls) == 2


def test_provider_timeout_propagates_for_the_fail_safe_wrapper_to_catch() -> None:
    grader = _grader(_RaisingLLMClient())

    with pytest.raises(TimeoutError):
        grader.grade("q", [_chunk()])


def test_grading_never_logs_the_raw_question_or_chunk_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm_client = _FakeLLMClient(
        payload={
            "coverage": 0.9,
            "chunks": [_chunk_payload(0, relevance=0.9, supports_question=True)],
        }
    )
    grader = _grader(llm_client)

    with caplog.at_level(logging.DEBUG):
        grader.grade("a very sensitive customer question", [_chunk("sensitive chunk text")])

    assert "a very sensitive customer question" not in caplog.text
    assert "sensitive chunk text" not in caplog.text


def test_cache_namespace_encodes_model_prompt_and_thresholds() -> None:
    grader = _grader(_FakeLLMClient(payload={"coverage": 0.0, "chunks": []}))

    assert grader.cache_namespace == (
        "grader=gpt-4o-mini:prompt=bank-policy-v1:correct=0.700:ambiguous=0.500:max_chunks=8"
    )


def test_thresholds_out_of_order_are_rejected_at_construction() -> None:
    with pytest.raises(ValueError):
        _grader(_FakeLLMClient(payload={}), correct_threshold=0.3, ambiguous_threshold=0.5)
