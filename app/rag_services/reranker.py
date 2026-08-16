from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.models.retrieved_chunk import RetrievedChunk

logger = logging.getLogger(__name__)

@dataclass
class ReRankedChunk:
    chunk: RetrievedChunk
    original_rank: int
    rerank_score: float| None

@dataclass
class ReRankOutcome:
    items: tuple[ReRankedChunk,...]
    backend: str
    applied: bool
    fallback: bool = False
    usage_tokens: int | None = None

@runtime_checkable
class ReRanker(Protocol):
    @property
    def name(self) -> str:
        """Human Readable backend name for metrics and response metadata"""
        ...

    @property
    def cache_namespace(self) -> str:
        """Stable identifier containing backend, model and implementation version."""
        ...

    def rerank(self,*,
               query: str,
               candidates: Sequence[RetrievedChunk],
               top_k: int) -> ReRankOutcome:
        ...

class NoOpReranker:
    @property
    def name(self) -> str:
        return "none"

    @property
    def cache_namespace(self) -> str:
        return "reranker:none:v1"

    def rerank(self,*,
               query: str,
               candidates: Sequence[RetrievedChunk],
               top_k: int) -> ReRankOutcome:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        items = tuple(
            ReRankedChunk(
                chunk=chunk,
                original_rank=rank,
                rerank_score=None,
            )
            for rank, chunk in enumerate(candidates[:top_k], start=1)
        )

        return ReRankOutcome(
            items = items,
            backend = self.name,
            applied=False,
        )


class FailOpenReranker:
    """Resilience decorator"""

    def __init__(self,
                 delegate: ReRanker,
                 fallback: ReRanker | None = None) -> None:
        self._delegate= delegate
        self._fallback = fallback


    @property
    def name(self) -> str:
        return self._delegate.name


    @property
    def cache_namespace(self) -> str:
        return f"fail-open:v1:{self._delegate.cache_namespace}"


    def rerank(self,
               *,
               query: str,
               candidates: Sequence[RetrievedChunk],
               top_k: int) -> ReRankOutcome:

        try:
            return self._delegate.rerank(
                query=query,
                candidates=candidates,
                top_k=top_k
            )
        except Exception:
            logger.exception(
                "Rerank failed, using retrieval ordering",
                extra={
                "reranker": self._delegate.name,
                "candidate_count": len(candidates),
                "top_k": top_k,
                },
            )

            if self._fallback is None:
                items = tuple(
                    ReRankedChunk(
                        chunk=chunk,
                        original_rank=rank,
                        rerank_score=None,
                    )
                    for rank, chunk in enumerate(candidates[:top_k], start=1)
                )

                return ReRankOutcome(
                    items=items,
                    backend=self._delegate.name,
                    applied=False,
                    fallback=True,
                )

            fallback = self._fallback.rerank(
                query=query,
                candidates=candidates,
                top_k=top_k,
            )

            return ReRankOutcome(
                items=fallback.items,
                backend=self._delegate.name,
                applied=False,
                fallback=True,
            )
