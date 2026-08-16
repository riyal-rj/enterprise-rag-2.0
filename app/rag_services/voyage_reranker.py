from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from typing import Any

from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.reranker import ReRankedChunk, ReRankOutcome


class VoyageReranker:
    def __init__(
        self, *, api_key: str, model_name: str, max_concurrency: int = 4, client: Any | None = None
    ) -> None:

        if not api_key:
            raise ValueError("api_key is required")

        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")

        if client is None:
            try:
                from voyageai import Client
            except ImportError:
                from voyageai.client import Client

            client = Client(api_key=api_key)

        self._client = client
        self._model_name = model_name
        self._request_slots = threading.BoundedSemaphore(max_concurrency)

    @property
    def name(self) -> str:
        return "voyage"

    @property
    def cache_namespace(self) -> str:
        return f"reranker:voyage:{self._model_name}:v1"

    def rerank(
        self, *, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> ReRankOutcome:

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not candidates:
            return ReRankOutcome(items=(), backend=self.name, applied=True)

        limit = min(top_k, len(candidates))

        with self._request_slots:
            response = self._client.rerank(
                query=query,
                documents=[candidate.text for candidate in candidates],
                model=self._model_name,
                top_k=limit,
                truncation=True,
            )

        seen_indexes: set[int] = set()
        ranked: list[ReRankedChunk] = []

        for result in response.results:
            index = int(result.index)
            score = float(result.relevance_score)

            if index < 0 or index >= len(candidates):
                raise RuntimeError("Voyage returned an invalid document index")

            if index in seen_indexes:
                raise RuntimeError("Voyage returned duplicate document index")

            if not math.isfinite(score):
                raise RuntimeError("Voyage returned an infinite score")

            seen_indexes.add(index)
            ranked.append(
                ReRankedChunk(chunk=candidates[index], original_rank=index + 1, rerank_score=score)
            )

        if len(ranked) != limit:
            raise RuntimeError(f"Voyage returned {len(ranked)} results; expected {limit}")

        def sort_key(item: ReRankedChunk) -> tuple[float, int]:
            score = item.rerank_score
            if score is None:
                raise RuntimeError("Voyage returned a null rerank score")
            return (-float(score), item.original_rank)

        ranked.sort(key=sort_key)

        return ReRankOutcome(
            items=tuple(ranked),
            backend=self.name,
            applied=True,
            usage_tokens=getattr(response, "total_tokens", None),
        )
