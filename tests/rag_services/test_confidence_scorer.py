from __future__ import annotations

import pytest

from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.confidence_scorer import compute_confidence, compute_confidence_breakdown


def test_no_chunks_scores_low() -> None:
    score = compute_confidence([], "I don't know based on the provided context.")

    assert score < 0.2


def test_strong_diverse_evidence_with_correct_citations_scores_high() -> None:
    chunks = [
        RetrievedChunk(text="refunds are issued within 30 days of purchase", source="a.pdf", score=0.9),
        RetrievedChunk(text="a 30 day refund window applies to all purchases", source="b.pdf", score=0.88),
        RetrievedChunk(text="refund requests must be made within 30 days", source="c.pdf", score=0.85),
    ]
    answer = "Refunds are available within 30 days of purchase [a.pdf][b.pdf][c.pdf]."

    score = compute_confidence(chunks, answer)

    assert score > 0.7


def test_weak_single_low_score_chunk_scores_lower_than_strong_evidence() -> None:
    weak_chunks = [RetrievedChunk(text="unrelated boilerplate text", source="z.pdf", score=0.3)]
    strong_chunks = [
        RetrievedChunk(text="refunds are issued within 30 days of purchase", source="a.pdf", score=0.9),
        RetrievedChunk(text="a 30 day refund window applies to all purchases", source="b.pdf", score=0.88),
    ]
    answer = "Refunds are available within 30 days of purchase [a.pdf][b.pdf]."

    weak_score = compute_confidence(weak_chunks, "Refunds are available within 30 days [a.pdf].")
    strong_score = compute_confidence(strong_chunks, answer)

    assert weak_score < strong_score


def test_citation_to_a_source_that_was_not_retrieved_is_penalized() -> None:
    chunks = [RetrievedChunk(text="refunds within 30 days", source="a.pdf", score=0.9)]

    grounded = compute_confidence(chunks, "Refunds are available within 30 days [a.pdf].")
    fabricated = compute_confidence(chunks, "Refunds are available within 30 days [made-up.pdf].")

    assert fabricated < grounded


def test_uncited_answer_scores_lower_than_cited_answer_given_same_evidence() -> None:
    chunks = [RetrievedChunk(text="refunds within 30 days of purchase", source="a.pdf", score=0.9)]

    cited = compute_confidence(chunks, "Refunds are available within 30 days of purchase [a.pdf].")
    uncited = compute_confidence(chunks, "Refunds are available within 30 days of purchase.")

    assert uncited < cited


def test_score_is_always_within_unit_interval() -> None:
    chunks = [RetrievedChunk(text="x" * 500, source="a.pdf", score=5.0)]

    score = compute_confidence(chunks, "y" * 500 + " [a.pdf]")

    assert 0.0 <= score <= 1.0


def test_multi_source_single_bracket_citation_parses_the_same_as_separate_brackets() -> None:
    chunks = [
        RetrievedChunk(text="refunds within 30 days", source="a.pdf", score=0.9),
        RetrievedChunk(text="digital goods excluded", source="b.pdf", score=0.85),
    ]

    single_bracket = compute_confidence(
        chunks, "Refunds within 30 days, except digital goods [a.pdf, b.pdf]."
    )
    separate_brackets = compute_confidence(
        chunks, "Refunds within 30 days, except digital goods [a.pdf][b.pdf]."
    )

    assert single_bracket == pytest.approx(separate_brackets)


def test_multi_source_bracket_does_not_get_penalized_as_a_fabricated_citation() -> None:
    chunks = [
        RetrievedChunk(text="refunds within 30 days", source="a.pdf", score=0.9),
        RetrievedChunk(text="digital goods excluded", source="b.pdf", score=0.85),
    ]

    breakdown = compute_confidence_breakdown(
        chunks, "Refunds within 30 days, except digital goods [a.pdf, b.pdf]."
    )

    assert breakdown.citation_precision == pytest.approx(1.0)


def test_short_refusal_scores_lower_than_a_long_answer_with_a_refusal_style_caveat() -> None:
    chunks = [
        RetrievedChunk(
            text="refunds within 30 days of purchase; no exceptions listed for digital goods",
            source="a.pdf",
            score=0.9,
        )
    ]
    short_refusal = "The provided context does not contain the answer."
    long_answer_with_caveat = (
        "Refunds are available within 30 days of purchase for most items [a.pdf]. "
        "The context does not contain a specific exception for digital goods, so "
        "confirm with support before assuming one applies."
    )

    short_score = compute_confidence(chunks, short_refusal)
    long_score = compute_confidence(chunks, long_answer_with_caveat)

    assert short_score < long_score


def test_breakdown_total_matches_weighted_sum_of_its_components() -> None:
    chunks = [RetrievedChunk(text="refunds within 30 days", source="a.pdf", score=0.9)]
    answer = "Refunds within 30 days [a.pdf]."

    breakdown = compute_confidence_breakdown(chunks, answer)

    expected = (
        0.30 * breakdown.evidence_coverage
        + 0.25 * breakdown.faithfulness
        + 0.20 * breakdown.retrieval_strength
        + 0.15 * breakdown.citation_precision
        + 0.10 * breakdown.answerability
    )
    assert breakdown.total == pytest.approx(expected)
    assert compute_confidence(chunks, answer) == breakdown.total
