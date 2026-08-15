from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from app.models.retrieved_chunk import RetrievedChunk
from app.repositories.vector_repository import VectorRepository
from app.rag_services.sparse_vector_service import SparseVectorIndex, fuse_rrf
class RetrievalStrategy(Protocol):
    """Contract implemented by every retrieval strategy."""

    @property
    def name(self) -> str:
        ...

    @property
    def cache_namespace(self) -> str:
        ...

    def retrieve(self,*,
                 query_text: str,
                 query_embedding: list[float],
                 top_k: int) -> list[RetrievedChunk]:
        ...

class DenseRetrievalStrategy:

    def __init__(self, vector_repository: VectorRepository) -> None:
        self._vector_repository = vector_repository

    @property
    def name(self) -> str:
        return "dense"

    @property
    def cache_namespace(self) -> str:
        return "dense:v1"

    def retrieve(self,*,
                 query_text: str,
                 query_embedding: list[float],
                 top_k:int) -> list[RetrievedChunk]:
        del query_text
        return self._vector_repository.search(query_embedding, top_k=top_k)
        
class HybridRetrievalStrategy:
    def __init__(self,
                 vector_repository: VectorRepository,
                *,
                rrf_k: int = 60,
                candidate_top_k: int = 20) -> None:
        self._vector_repository = vector_repository
        self._rrf_k = rrf_k
        self._candidate_top_k = candidate_top_k

    @property
    def name(self) -> str:
        return "hybrid"

    @property
    def cache_namespace(self) -> str:
        return "hybrid:v1"

    def retrieve(self,*,
                 query_text: str,
                 query_embedding: list[float],
                 top_k:int) -> list[RetrievedChunk]:
        candidate_top_k = max(top_k, self._candidate_top_k)
        return self.hybrid_search(query_embedding=query_embedding,
                                  query_text=query_text, 
                                  top_k=top_k,
                                  rrf_k=self._rrf_k,
                                  sparse_top_k=candidate_top_k)
    
    def hybrid_search(self,
                      query_embedding: list[float],
                      query_text: str,
                      top_k: int = 5,
                      rrf_k: int = 60,
                      sparse_top_k: int = 20) -> list[RetrievedChunk]:
        candidate_top_k = max(top_k, sparse_top_k)

        dense_results = self._vector_repository.search(
            query_embedding,
            top_k=candidate_top_k
        )

        sparse_results = self._build_sparse_index().search(
            query_text,
            top_k=candidate_top_k
        )

        fused_results = fuse_rrf([dense_results, sparse_results], rrf_k=rrf_k)

        max_rrf_score = 2.0 / (rrf_k + 1)

        normalized_results = [
            replace(
                chunk,
                score=min(chunk.score/ max_rrf_score, 1.0)
            )
            for chunk in fused_results
        ]

        return normalized_results[:top_k]

    def _build_sparse_index(self) -> SparseVectorIndex:
        documents = self._vector_repository.scroll_all_chunks()
        sparse_index = SparseVectorIndex()
        sparse_index.fit(documents)
        return sparse_index
