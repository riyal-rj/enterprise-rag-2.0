from __future__ import annotations

from app.models.retrieved_chunk import RetrievedChunk
from app.repositories.vector_repository import VectorRepository
from app.rag_services.sparse_vector_service import SparseVectorIndex, fuse_rrf


class HybridRetrievalService:
    def __init__(self, vector_repository: VectorRepository) -> None:
        self._vector_repository = vector_repository

    def sparse_search(self, query_text: str, top_k: int = 5) -> list[RetrievedChunk]:
        return self._build_sparse_index().search(query_text, top_k=top_k)

    def _build_sparse_index(self) -> SparseVectorIndex:
        documents = self._vector_repository.scroll_all_chunks()
        sparse_index = SparseVectorIndex()
        sparse_index.fit(documents)
        return sparse_index

    def hubrid_search(self,query_embedding: list[float],
                           query_text: str,
                           top_k: int = 5,
                           rrf_k: int = 5,
                           sparse_top_k: int = 20) -> list[RetrievedChunk] :
        dense_results = self._vector_repository.search(query_embedding, top_k = sparse_top_k)
        sparse_results = self._build_sparse_index().search(query_text, top_k=rrf_k)
        fused_results = fuse_rrf([dense_results, sparse_results], rrf_k = rrf_k)
        return fused_results[: top_k]