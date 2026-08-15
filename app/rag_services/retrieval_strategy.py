"""Retrieval strategies (Strategy pattern) - dense, sparse, and hybrid.

Fusion for hybrid is server-side, via Qdrant's Query API (prefetch + RRF,
see ``app.repositories.vector_repository.QdrantVectorRepository.search_hybrid``)
- there's no in-process TF-IDF fitting or fusion here anymore (that was
``app.rag_services.sparse_vector_service``, now deleted).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from app.core.llm.sparse_embedding_client import SparseEmbeddingClient
from app.models.retrieved_chunk import RetrievedChunk
from app.repositories.vector_repository import VectorRepository


class RetrievalStrategy(Protocol):
    """Contract implemented by every retrieval strategy."""

    @property
    def name(self) -> str: ...

    @property
    def cache_namespace(self) -> str: ...

    def retrieve(
        self, *, query_text: str, query_embedding: list[float], top_k: int
    ) -> list[RetrievedChunk]: ...


class DenseRetrievalStrategy:
    def __init__(self, vector_repository: VectorRepository) -> None:
        self._vector_repository = vector_repository

    @property
    def name(self) -> str:
        return "dense"

    @property
    def cache_namespace(self) -> str:
        return "dense:v1"

    def retrieve(
        self, *, query_text: str, query_embedding: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        del query_text
        return self._vector_repository.search_dense(query_embedding, top_k=top_k)


class SparseRetrievalStrategy:
    """Pure BM25 lexical search - no dense embedding, no fusion.

    Matches ``app.eval.profiles``' ``search_mode="sparse"``/``sparse_only``
    profile: a first-class production mode for comparing against dense and
    hybrid, not just an eval-only concept.
    """

    def __init__(
        self, vector_repository: VectorRepository, sparse_embedding_client: SparseEmbeddingClient
    ) -> None:
        self._vector_repository = vector_repository
        self._sparse_embedding_client = sparse_embedding_client

    @property
    def name(self) -> str:
        return "sparse"

    @property
    def cache_namespace(self) -> str:
        return "sparse:v1"

    def retrieve(
        self, *, query_text: str, query_embedding: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        del query_embedding
        query_sparse = self._sparse_embedding_client.embed_query(query_text)
        return self._vector_repository.search_sparse(query_sparse, top_k=top_k)


class HybridRetrievalStrategy:
    def __init__(
        self,
        vector_repository: VectorRepository,
        sparse_embedding_client: SparseEmbeddingClient,
        *,
        rrf_k: int = 60,
        candidate_top_k: int = 20,
    ) -> None:
        self._vector_repository = vector_repository
        self._sparse_embedding_client = sparse_embedding_client
        self._rrf_k = rrf_k
        self._candidate_top_k = candidate_top_k

    @property
    def name(self) -> str:
        return "hybrid"

    @property
    def cache_namespace(self) -> str:
        # v2: the underlying implementation changed (Qdrant-native fusion +
        # BM25 instead of in-process TF-IDF) - bumped so a cached answer
        # from the old pipeline can't be served under the new one's key.
        return "hybrid:v2"

    def retrieve(
        self, *, query_text: str, query_embedding: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        candidate_top_k = max(top_k, self._candidate_top_k)
        return self.hybrid_search(
            query_embedding=query_embedding,
            query_text=query_text,
            top_k=top_k,
            rrf_k=self._rrf_k,
            candidate_top_k=candidate_top_k,
        )

    def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 5,
        rrf_k: int = 60,
        candidate_top_k: int = 20,
    ) -> list[RetrievedChunk]:
        candidate_top_k = max(top_k, candidate_top_k)
        query_sparse = self._sparse_embedding_client.embed_query(query_text)

        fused_results = self._vector_repository.search_hybrid(
            query_embedding=query_embedding,
            query_sparse=query_sparse,
            top_k=top_k,
            candidate_top_k=candidate_top_k,
            rrf_k=rrf_k,
        )

        # Qdrant's RRF uses 0-indexed ranks (score = sum of 1/(k+rank),
        # rank starting at 0 for the top result in each prefetch branch -
        # confirmed empirically against the live server, not assumed) - a
        # chunk ranked first in both the dense and sparse branches gets the
        # maximum possible score, 2/k. Normalizing against that keeps
        # hybrid scores on the same [0,1] scale confidence_scorer's
        # _RELEVANCE_FLOOR expects (calibrated for dense cosine similarity).
        max_rrf_score = 2.0 / rrf_k
        return [
            replace(chunk, score=min(chunk.score / max_rrf_score, 1.0)) for chunk in fused_results
        ]
