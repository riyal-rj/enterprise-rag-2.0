from __future__ import annotations

from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.claim_checker import find_unsupported_claims


def test_generalized_claim_not_grounded_in_context_is_flagged() -> None:
    chunks = [
        RetrievedChunk(
            text="Outgoing debits are frozen once a hold is placed on the account.",
            source="policy.pdf",
            score=0.9,
        )
    ]
    answer = "The account cannot send or receive any transactions while the hold is active."

    flagged = find_unsupported_claims(answer, chunks)

    assert flagged
    assert "cannot send or receive" in flagged[0]


def test_claim_closely_grounded_in_context_is_not_flagged() -> None:
    chunks = [
        RetrievedChunk(
            text="Outgoing debits are frozen once a hold is placed on the account.",
            source="policy.pdf",
            score=0.9,
        )
    ]
    answer = "Outgoing debits are frozen once a hold is placed on the account [policy.pdf]."

    flagged = find_unsupported_claims(answer, chunks)

    assert flagged == []


def test_sentence_without_absolute_claim_language_is_not_flagged() -> None:
    chunks = [RetrievedChunk(text="refunds within 30 days", source="a.pdf", score=0.9)]
    answer = "Refunds are typically available within 30 days of purchase [a.pdf]."

    assert find_unsupported_claims(answer, chunks) == []


def test_no_chunks_flags_any_absolute_claim() -> None:
    answer = "Customers must always report suspicious activity within 24 hours."

    flagged = find_unsupported_claims(answer, [])

    assert flagged == [answer]
