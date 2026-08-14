"""Sparse+dense hybrid retrieval, composed over ``VectorRepository``.

Kept separate from ``VectorRepository`` (pure Qdrant persistence): sparse
search and RRF fusion are retrieval *strategy* concerns (see
``RAGFeatureSettings.hybrid_search_enabled``/``rrf_k``), not vector-store
persistence, so this composes over the repository rather than living
inside it - consistent with the reranker/reflection Strategy split
described in ``app.core.config.rag_features``.
"""

from __future__ import annotations

from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.sparse_vector_service import SparseVectorIndex, fuse_rrf
from app.repositories.vector_repository import VectorRepository


class HybridRetrievalService:
    def __init__(self, vector_repository: VectorRepository) -> None:
        self._vector_repository = vector_repository

    def sparse_search(self, query_text: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Pure sparse search using TF-IDF (no dense embeddings, no fusion)."""
        return self._build_sparse_index().search(query_text, top_k=top_k)

    def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 5,
        rrf_k: int = 60,
        sparse_top_k: int = 20,
    ) -> list[RetrievedChunk]:
        dense_results = self._vector_repository.search(query_embedding, top_k=sparse_top_k)
        sparse_results = self._build_sparse_index().search(query_text, top_k=sparse_top_k)
        fused_results = fuse_rrf([dense_results, sparse_results], rrf_k=rrf_k)
        return fused_results[:top_k]

    def _build_sparse_index(self) -> SparseVectorIndex:
        documents = self._vector_repository.scroll_all_chunks()
        sparse_index = SparseVectorIndex()
        sparse_index.fit(documents)
        return sparse_index
