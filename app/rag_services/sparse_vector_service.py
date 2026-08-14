from __future__ import annotations

import threading
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from app.models.retrieved_chunk import RetrievedChunk

class SparseVectorIndex:

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

            texts = [doc.get("text","") for doc in documents]
            try:
                matrix = self._vectorizer.fit_transform(texts)
            except ValueError:
                return
            if matrix.shape[1] > 0 :
                self._matrix = matrix



    def search(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        with self._lock:
            if self._matrix is None:
                return []

            query_vector = self._vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self._matrix).flatten()
            top_indices = similarities.argsort()[::-1][:top_k]

            results: list[RetrievedChunk] = []
            for index  in top_indices:
                score = float(similarities[index])
                if score <= 0:
                    continue
                doc = self._documents[index]
                results.append(
                    RetrievedChunk(
                        text=doc.get("text",""),
                        source=doc.get("source",""),
                        score=score
                    )
                )

            return results


def fuse_rrf(result_lists: list[list[RetrievedChunk]], rrf_k: int = 60) -> list[RetrievedChunk]:
    scores: dict[tuple[str, str], float] = {}
    chunks_by_key: dict[tuple[str, str], RetrievedChunk] = {}


    for result_list in result_lists:
        for rank, chunk in enumerate(result_list, start = 1):
            key = (chunk.source, chunk.text)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            chunks_by_key.setdefault(key, chunk)

    ranked_keys = sorted(scores, key = lambda key: scores[key], reverse = True)
    return [
        RetrievedChunk(
            text = key[1],
            source = key[0],
            score = scores[key]
        )
        for key in ranked_keys
    ]
