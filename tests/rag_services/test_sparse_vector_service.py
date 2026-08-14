from __future__ import annotations

from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.sparse_vector_service import SparseVectorIndex, fuse_rrf

# Real TfidfVectorizer/cosine_similarity throughout - no fakes, since this
# class is a thin wrapper around them and faking sklearn would just test
# the fakes, not the actual TF-IDF behavior.

_DOCUMENTS = [
    {"id": "1", "text": "refund policy for international wire transfers", "source": "a.pdf"},
    {"id": "2", "text": "branch opening hours and public holidays", "source": "b.pdf"},
]


def test_search_before_fit_returns_empty_list() -> None:
    index = SparseVectorIndex()

    assert index.search("refund policy") == []


def test_fit_then_search_ranks_by_relevance() -> None:
    index = SparseVectorIndex()
    index.fit(_DOCUMENTS)

    results = index.search("what is the refund policy", top_k=5)

    assert len(results) == 1  # the holidays doc scores exactly 0 and is excluded
    assert results[0].source == "a.pdf"
    assert results[0].text == _DOCUMENTS[0]["text"]
    assert results[0].score > 0


def test_search_respects_top_k() -> None:
    index = SparseVectorIndex()
    index.fit(
        [
            {"id": "1", "text": "refund policy overseas transactions", "source": "a.pdf"},
            {"id": "2", "text": "refund policy domestic transactions", "source": "b.pdf"},
        ]
    )

    results = index.search("refund policy transactions", top_k=1)

    assert len(results) == 1


def test_fit_empty_documents_results_in_empty_search() -> None:
    index = SparseVectorIndex()
    index.fit([])

    assert index.search("anything") == []


def test_fit_all_stop_word_documents_does_not_raise() -> None:
    """TfidfVectorizer raises ValueError on an empty vocabulary; fit()
    must swallow it rather than propagate, since a caller building the
    index from a live scroll can't guarantee every document has content
    words."""
    index = SparseVectorIndex()

    index.fit([{"id": "1", "text": "the a an", "source": "a.pdf"}])

    assert index.search("anything") == []


def test_refit_replaces_previous_index() -> None:
    index = SparseVectorIndex()
    index.fit(_DOCUMENTS)
    assert index.search("refund policy") != []

    index.fit([])

    assert index.search("refund policy") == []


def test_fuse_rrf_ranks_chunk_appearing_in_both_lists_first() -> None:
    shared = RetrievedChunk(text="shared", source="a.pdf", score=0.5)
    dense_only = RetrievedChunk(text="dense only", source="b.pdf", score=0.9)
    sparse_only = RetrievedChunk(text="sparse only", source="c.pdf", score=0.9)

    fused = fuse_rrf([[shared, dense_only], [shared, sparse_only]], rrf_k=60)

    assert fused[0].text == "shared"


def test_fuse_rrf_distinguishes_same_text_from_different_sources() -> None:
    chunk_a = RetrievedChunk(text="same boilerplate text", source="a.pdf", score=0.5)
    chunk_b = RetrievedChunk(text="same boilerplate text", source="b.pdf", score=0.5)

    fused = fuse_rrf([[chunk_a], [chunk_b]])

    assert {c.source for c in fused} == {"a.pdf", "b.pdf"}
    assert len(fused) == 2


def test_fuse_rrf_empty_lists_returns_empty() -> None:
    assert fuse_rrf([]) == []
    assert fuse_rrf([[], []]) == []


def test_fuse_rrf_replaces_original_score_with_fused_score() -> None:
    chunk = RetrievedChunk(text="hello", source="a.pdf", score=0.9)

    fused = fuse_rrf([[chunk]], rrf_k=60)

    assert fused[0].score == 1.0 / 61  # rank 1 in a single list
