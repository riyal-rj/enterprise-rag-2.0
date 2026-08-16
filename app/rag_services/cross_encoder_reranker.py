from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence
from typing import Any

from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.reranker import ReRankedChunk, ReRankOutcome


class LocalCrossEncoderReranker:
    def __init__(
        self,
        *,
        model_name: str,
        batch_size: int = 16,
        max_concurrency: int = 1,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")

        self._model_name = model_name
        self._batch_size = batch_size
        self._max_concurrency = max_concurrency
        self._model_factory = model_factory
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._inference_slots = threading.BoundedSemaphore(max_concurrency)

    @property
    def name(self) -> str:
        return "local-cross-encoder"

    @property
    def cache_namespace(self) -> str:
        return f"reranker:local:{self._model_name}:v1"

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is None:
                if self._model_factory is not None:
                    self._model = self._model_factory(self._model_name)
                else:
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self, *, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> ReRankOutcome:

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not candidates:
            return ReRankOutcome(items=(), backend=self.name, applied=True)

        pairs = [(query, candidate.text) for candidate in candidates]

        with self._inference_slots:
            scores = self._get_model().predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
            )

        if len(scores) != len(candidates):
            raise RuntimeError("CrossEncoder returned an unexpected number of scores")

        scored_candidates: list[tuple[int, RetrievedChunk, float]] = []
        for zero_based_rank, (candidate, raw_score) in enumerate(
            zip(candidates, scores, strict=True)
        ):
            score = float(raw_score)

            if not math.isfinite(score):
                raise RuntimeError("CrossEncoder returned an infinite score")

            scored_candidates.append((zero_based_rank, candidate, score))

        scored_candidates.sort(key=lambda item: (-item[2], item[0]))

        items = tuple(
            ReRankedChunk(
                chunk=candidate,
                original_rank=original_rank + 1,
                rerank_score=score,
            )
            for original_rank, candidate, score in scored_candidates[
                : min(top_k, len(scored_candidates))
            ]
        )

        return ReRankOutcome(items=items, backend=self.name, applied=True)
