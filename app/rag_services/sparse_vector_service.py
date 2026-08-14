"""Sparse (TF-IDF) lexical search and Reciprocal Rank Fusion.

``SparseVectorIndex`` is the sparse counterpart to ``VectorRepository``'s
dense (embedding) search: lexical/keyword matching via TF-IDF cosine
similarity, no embedding calls involved. ``fuse_rrf`` combines any number
of ranked result lists (e.g. dense + sparse) into one - see
``RAGFeatureSettings.rrf_k`` and ``HybridRetrievalService``.
"""

from __future__ import annotations

import threading
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models.retrieved_chunk import RetrievedChunk


class SparseVectorIndex:
    """In-memory TF-IDF index, lock-guarded so it's safe to share.

    ``fit``/``search`` are public methods on what could become a
    longer-lived, concurrently-accessed instance (same shared-mutable-
    state-plus-lock shape as ``QueryCacheService`` elsewhere in this
    codebase) - the lock costs nothing single-threaded, and means this
    class doesn't need revisiting if it's later cached instead of
    rebuilt per search.
    """

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._documents: list[dict[str, str]] = []
        self._matrix: Any | None = None
        self._lock = threading.RLock()

    def fit(self, documents: list[dict[str, str]]) -> None:
        with self._lock:
            self._documents = documents
            self._matrix = None
            if not documents:
                return

            texts = [doc.get("text", "") for doc in documents]
            try:
                matrix = self._vectorizer.fit_transform(texts)
            except ValueError:
                # Empty vocabulary - every text was blank or only stop words.
                return
            if matrix.shape[1] > 0:
                self._matrix = matrix

    def search(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        """Return up to ``top_k`` chunks by TF-IDF cosine similarity."""
        with self._lock:
            if self._matrix is None:
                return []

            query_vector = self._vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self._matrix).flatten()
            top_indices = similarities.argsort()[::-1][:top_k]

            results: list[RetrievedChunk] = []
            for index in top_indices:
                score = float(similarities[index])
                if score <= 0:
                    continue
                doc = self._documents[index]
                results.append(
                    RetrievedChunk(
                        text=doc.get("text", ""), source=doc.get("source", ""), score=score
                    )
                )
            return results


def fuse_rrf(result_lists: list[list[RetrievedChunk]], rrf_k: int = 60) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion over any number of ranked result lists.

    Each chunk's fused score is the sum, across every list it appears in,
    of ``1 / (rrf_k + rank)`` (rank is 1-indexed) - deliberately ignoring
    the lists' own raw scores, which aren't on comparable scales (cosine
    similarity vs. TF-IDF similarity). Chunks are identified by
    ``(source, text)``, not text alone: two different source documents
    can legitimately share identical boilerplate text (e.g. a shared
    disclaimer paragraph), and collapsing those into one entry would
    silently drop a real hit and misattribute its source.
    """
    scores: dict[tuple[str, str], float] = {}
    chunks_by_key: dict[tuple[str, str], RetrievedChunk] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list, start=1):
            key = (chunk.source, chunk.text)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            chunks_by_key.setdefault(key, chunk)

    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [RetrievedChunk(text=key[1], source=key[0], score=scores[key]) for key in ranked_keys]
