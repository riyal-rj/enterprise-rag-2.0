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


def test_hybrid_mode_treats_all_returned_chunks_as_supporting_evidence() -> None:
    """A hybrid RRF-normalized score isn't on the cosine scale
    _RELEVANCE_FLOOR was calibrated for (see the confidence_scorer module
    docstring) - evidence_coverage must not gate non-dense scores against
    it the same way it gates dense cosine similarity."""
    chunks = [RetrievedChunk(text="refunds within 30 days", source="a.pdf", score=0.1)]

    hybrid = compute_confidence_breakdown(
        chunks, "Refunds within 30 days [a.pdf].", retrieval_mode="hybrid"
    )
    dense = compute_confidence_breakdown(
        chunks, "Refunds within 30 days [a.pdf].", retrieval_mode="dense"
    )

    assert hybrid.evidence_coverage > 0.0
    assert dense.evidence_coverage == 0.0  # below _RELEVANCE_FLOOR


def test_hybrid_mode_retrieval_strength_is_neutral_regardless_of_score() -> None:
    """No calibrated absolute-relevance signal exists yet for a
    rank-derived hybrid score - retrieval_strength must stay a fixed
    neutral value rather than reading a false confidence signal into it."""
    low = compute_confidence_breakdown(
        [RetrievedChunk(text="x", source="a.pdf", score=0.05)],
        "x [a.pdf]",
        retrieval_mode="hybrid",
    )
    high = compute_confidence_breakdown(
        [RetrievedChunk(text="x", source="a.pdf", score=0.95)],
        "x [a.pdf]",
        retrieval_mode="hybrid",
    )

    assert low.retrieval_strength == pytest.approx(0.5)
    assert high.retrieval_strength == pytest.approx(0.5)


def test_dense_mode_retrieval_strength_still_varies_with_score() -> None:
    """The dense-mode calibration must be unaffected by the hybrid/sparse
    neutral fallback - this is the default and the only mode with a
    genuinely calibrated score."""
    low = compute_confidence_breakdown(
        [RetrievedChunk(text="x", source="a.pdf", score=0.1)], "x [a.pdf]", retrieval_mode="dense"
    )
    high = compute_confidence_breakdown(
        [RetrievedChunk(text="x", source="a.pdf", score=0.9)], "x [a.pdf]", retrieval_mode="dense"
    )

    assert low.retrieval_strength < high.retrieval_strength


def test_retrieval_mode_defaults_to_dense() -> None:
    chunks = [RetrievedChunk(text="x", source="a.pdf", score=0.1)]

    explicit_dense = compute_confidence_breakdown(chunks, "x [a.pdf]", retrieval_mode="dense")
    default = compute_confidence_breakdown(chunks, "x [a.pdf]")

    assert default == explicit_dense


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


def test_retrieval_strength_defaults_to_reading_chunks_order_when_not_given() -> None:
    """Without an explicit retrieval_ordered_chunks, retrieval_strength
    must keep reading position 0 of ``chunks`` itself - the pre-reranking
    behavior every non-reranking caller still relies on."""
    chunks = [
        RetrievedChunk(text="a", source="a.pdf", score=0.95),
        RetrievedChunk(text="b", source="b.pdf", score=0.20),
    ]
    answer = "a [a.pdf]"

    default = compute_confidence_breakdown(chunks, answer)
    explicit = compute_confidence_breakdown(chunks, answer, retrieval_ordered_chunks=chunks)

    assert default.retrieval_strength == explicit.retrieval_strength


def test_retrieval_strength_uses_retrieval_ordered_chunks_not_reranked_order() -> None:
    """A reranker can promote a chunk with a weak retrieval score to
    position 0 while leaving .score (the original retrieval score)
    untouched. retrieval_strength must reflect the strength of the
    original retrieval's top hit, not whichever chunk the reranker put
    first - otherwise a strong candidate pool the reranker buried reads
    as a weak one, and vice versa."""
    retrieval_order = [
        RetrievedChunk(text="strongest retrieval hit", source="a.pdf", score=0.95),
        RetrievedChunk(text="weak retrieval hit", source="b.pdf", score=0.15),
    ]
    # Reranker promoted the weak-retrieval-score chunk to position 0.
    reranked_order = [retrieval_order[1], retrieval_order[0]]
    answer = "weak retrieval hit [b.pdf]"

    naive = compute_confidence_breakdown(reranked_order, answer)
    corrected = compute_confidence_breakdown(
        reranked_order, answer, retrieval_ordered_chunks=retrieval_order
    )

    # Naive (reads reranked_order[0].score=0.15) must score retrieval
    # strength lower than corrected (reads retrieval_order[0].score=0.95).
    assert naive.retrieval_strength < corrected.retrieval_strength


def test_retrieval_ordered_chunks_only_affects_retrieval_strength() -> None:
    """The other four components must still read the actual (reranked)
    chunks passed as `chunks` - they describe what the LLM saw and cited,
    which retrieval_ordered_chunks must not silently override."""
    chunks = [RetrievedChunk(text="refunds within 30 days", source="a.pdf", score=0.9)]
    different_ordering_chunks = [RetrievedChunk(text="unrelated", source="z.pdf", score=0.1)]
    answer = "Refunds within 30 days [a.pdf]."

    breakdown = compute_confidence_breakdown(
        chunks, answer, retrieval_ordered_chunks=different_ordering_chunks
    )
    baseline = compute_confidence_breakdown(chunks, answer)

    assert breakdown.evidence_coverage == baseline.evidence_coverage
    assert breakdown.faithfulness == baseline.faithfulness
    assert breakdown.citation_precision == baseline.citation_precision
    assert breakdown.answerability == baseline.answerability
