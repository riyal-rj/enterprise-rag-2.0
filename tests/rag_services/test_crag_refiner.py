"""Tests for ExtractiveKnowledgeRefiner."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.llm.chat_client import StructuredLLMResponse, TokenUsage
from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.crag.crag import (
    ChunkGrade,
    CRAGDecision,
    EvidenceOrigin,
    RetrievalGrade,
    WebEvidence,
)
from app.rag_services.crag.crag_refiner import ExtractiveKnowledgeRefiner


class _FakeLLMClient:
    def __init__(self, *, payload: dict[str, object] | None = None, usage_tokens: int = 3) -> None:
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
        self.calls.append({"user_message": user_message})
        assert self._payload is not None
        value = response_model(**self._payload)
        return StructuredLLMResponse(value=value, usage=TokenUsage(total_tokens=self._usage_tokens))


def _refiner(llm_client: object, **overrides: object) -> ExtractiveKnowledgeRefiner:
    defaults: dict[str, object] = dict(
        llm_client=llm_client,
        model="gpt-4o-mini",
        min_relevance=0.5,
        max_documents=5,
        max_sentences_per_document=80,
        timeout_seconds=10.0,
        max_completion_tokens=800,
        prompt_version="bank-policy-v1",
    )
    defaults.update(overrides)
    return ExtractiveKnowledgeRefiner(**defaults)  # type: ignore[arg-type]


def _grade(
    *items: ChunkGrade, coverage: float = 0.9, decision: CRAGDecision = CRAGDecision.CORRECT
) -> RetrievalGrade:
    return RetrievalGrade(decision=decision, coverage=coverage, chunks=tuple(items))


def test_only_selected_sentences_appear_verbatim() -> None:
    chunk = RetrievedChunk(
        text="Refunds are issued within 30 days. Fees are non-refundable after use.",
        source="a.pdf",
        score=0.9,
        page_number=2,
    )
    grade = _grade(
        ChunkGrade(index=0, relevance=0.9, supports_question=True, reason_code="directly_relevant")
    )
    llm_client = _FakeLLMClient(
        payload={"selections": [{"document_index": 0, "sentence_indices": [0]}]}
    )
    refiner = _refiner(llm_client)

    evidence, tokens = refiner.refine_local("q", [chunk], grade)

    assert len(evidence) == 1
    assert evidence[0].text == "Refunds are issued within 30 days."
    assert "non-refundable" not in evidence[0].text
    assert evidence[0].source == "a.pdf"
    assert evidence[0].page_number == 2
    assert evidence[0].origin is EvidenceOrigin.POLICY
    assert tokens == 3


def test_conditions_negations_and_thresholds_remain_verbatim() -> None:
    """Extractive selection must never rewrite - a conditional/negated
    sentence must reach evidence exactly as written, not paraphrased."""
    chunk = RetrievedChunk(
        text="A transaction is not permitted unless the customer completes EDD above $50,000.",
        source="a.pdf",
        score=0.9,
        page_number=1,
    )
    grade = _grade(
        ChunkGrade(index=0, relevance=0.9, supports_question=True, reason_code="directly_relevant")
    )
    llm_client = _FakeLLMClient(
        payload={"selections": [{"document_index": 0, "sentence_indices": [0]}]}
    )
    refiner = _refiner(llm_client)

    evidence, _ = refiner.refine_local("q", [chunk], grade)

    assert evidence[0].text == (
        "A transaction is not permitted unless the customer completes EDD above $50,000."
    )


def test_empty_selection_returns_empty_evidence() -> None:
    chunk = RetrievedChunk(text="irrelevant text here.", source="a.pdf", score=0.9)
    grade = _grade(
        ChunkGrade(index=0, relevance=0.9, supports_question=True, reason_code="directly_relevant")
    )
    llm_client = _FakeLLMClient(payload={"selections": []})
    refiner = _refiner(llm_client)

    evidence, tokens = refiner.refine_local("q", [chunk], grade)

    assert evidence == ()
    assert tokens == 3
    # No documents at all (nothing cleared the relevance bar) must not even
    # call the LLM.
    llm_client2 = _FakeLLMClient(payload={"selections": []})
    refiner2 = _refiner(llm_client2)
    below_bar_grade = _grade(
        ChunkGrade(index=0, relevance=0.1, supports_question=False, reason_code="irrelevant")
    )
    evidence2, tokens2 = refiner2.refine_local("q", [chunk], below_bar_grade)
    assert evidence2 == ()
    assert tokens2 == 0
    assert llm_client2.calls == []


def test_unknown_document_index_is_rejected() -> None:
    chunk = RetrievedChunk(text="text.", source="a.pdf", score=0.9)
    grade = _grade(
        ChunkGrade(index=0, relevance=0.9, supports_question=True, reason_code="directly_relevant")
    )
    llm_client = _FakeLLMClient(
        payload={"selections": [{"document_index": 5, "sentence_indices": [0]}]}
    )
    refiner = _refiner(llm_client)

    with pytest.raises(ValueError, match="unknown document index"):
        refiner.refine_local("q", [chunk], grade)


def test_unknown_sentence_index_is_rejected() -> None:
    chunk = RetrievedChunk(text="Only one sentence here.", source="a.pdf", score=0.9)
    grade = _grade(
        ChunkGrade(index=0, relevance=0.9, supports_question=True, reason_code="directly_relevant")
    )
    llm_client = _FakeLLMClient(
        payload={"selections": [{"document_index": 0, "sentence_indices": [9]}]}
    )
    refiner = _refiner(llm_client)

    with pytest.raises(ValueError, match="unknown sentence index"):
        refiner.refine_local("q", [chunk], grade)


def test_duplicate_sentence_indices_are_rejected() -> None:
    chunk = RetrievedChunk(text="One. Two.", source="a.pdf", score=0.9)
    grade = _grade(
        ChunkGrade(index=0, relevance=0.9, supports_question=True, reason_code="directly_relevant")
    )
    llm_client = _FakeLLMClient(
        payload={"selections": [{"document_index": 0, "sentence_indices": [0, 0]}]}
    )
    refiner = _refiner(llm_client)

    with pytest.raises(ValidationError):
        refiner.refine_local("q", [chunk], grade)


def test_refine_web_preserves_source_url_and_timestamp() -> None:
    result = WebEvidence(
        title="RBI Circular on KYC",
        text="Periodic KYC updation is mandatory for high-risk customers.",
        canonical_url="https://rbi.org.in/circular/123",
        domain="rbi.org.in",
        retrieved_at_iso="2026-01-01T00:00:00+00:00",
        score=0.8,
    )
    llm_client = _FakeLLMClient(
        payload={"selections": [{"document_index": 0, "sentence_indices": [0]}]}
    )
    refiner = _refiner(llm_client)

    evidence, _ = refiner.refine_web("q", [result])

    assert len(evidence) == 1
    item = evidence[0]
    assert item.text == "Periodic KYC updation is mandatory for high-risk customers."
    assert item.source == "RBI Circular on KYC"
    assert item.canonical_url == "https://rbi.org.in/circular/123"
    assert item.retrieved_at_iso == "2026-01-01T00:00:00+00:00"
    assert item.origin is EvidenceOrigin.REGULATORY_WEB
    assert item.page_number is None


def test_refine_web_unknown_document_index_is_rejected() -> None:
    result = WebEvidence(
        title="t",
        text="text.",
        canonical_url="https://rbi.org.in/x",
        domain="rbi.org.in",
        retrieved_at_iso="2026-01-01T00:00:00+00:00",
        score=0.5,
    )
    llm_client = _FakeLLMClient(
        payload={"selections": [{"document_index": 3, "sentence_indices": [0]}]}
    )
    refiner = _refiner(llm_client)

    with pytest.raises(ValueError, match="unknown web document index"):
        refiner.refine_web("q", [result])


def test_relevance_floor_and_max_documents_cap_what_reaches_the_llm() -> None:
    chunks = [
        RetrievedChunk(text="A relevant sentence one.", source="a.pdf", score=0.9),
        RetrievedChunk(text="A relevant sentence two.", source="b.pdf", score=0.8),
        RetrievedChunk(text="An irrelevant sentence.", source="c.pdf", score=0.1),
    ]
    grade = _grade(
        ChunkGrade(index=0, relevance=0.9, supports_question=True, reason_code="directly_relevant"),
        ChunkGrade(index=1, relevance=0.6, supports_question=True, reason_code="partial_support"),
        ChunkGrade(index=2, relevance=0.1, supports_question=False, reason_code="irrelevant"),
    )
    llm_client = _FakeLLMClient(payload={"selections": []})
    refiner = _refiner(llm_client, max_documents=1)

    refiner.refine_local("q", chunks, grade)

    sent_payload = llm_client.calls[0]["user_message"]
    assert "sentence one" in str(sent_payload)
    assert "sentence two" not in str(sent_payload)
    assert "irrelevant sentence" not in str(sent_payload)
