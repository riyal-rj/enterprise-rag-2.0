"""BM25 sparse embeddings via FastEmbed - local, no API calls.

Stateless: unlike a classic ``TfidfVectorizer`` (see the now-deleted
``app.rag_services.sparse_vector_service.SparseVectorIndex``), FastEmbed's
``Qdrant/bm25`` model needs no corpus-fitting step - term weights are
computed per-text via hashed vocabulary indices, and real corpus-aware IDF
weighting happens server-side via the Qdrant collection's sparse vector
``Modifier.IDF`` (see ``app.repositories.vector_repository``), not here.

Document and query encoding are genuinely different for BM25: document
vectors carry term-frequency + length-normalization weights (``embed()``),
while query vectors give every unique query term a flat weight of 1.0
(``query_embed()``) - the two must not be used interchangeably.
"""

from __future__ import annotations

from typing import Protocol

from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector


class SparseEmbeddingClient(Protocol):
    """Contract for a sparse (lexical) embedding provider."""

    def embed_documents(self, texts: list[str]) -> list[SparseVector]: ...

    def embed_query(self, text: str) -> SparseVector: ...


class FastEmbedSparseClient:
    """:class:`SparseEmbeddingClient` backed by FastEmbed's BM25 model.

    The model name is pinned explicitly (not left to FastEmbed's default)
    so ingestion and query-time encoding can't silently drift apart if
    FastEmbed changes its default model in a future release.

    Model loading is deferred to the first actual embed call, not
    ``__init__``: this client is always constructed by
    ``app.api.deps.get_retrieval_strategies`` regardless of
    ``HYBRID_SEARCH_ENABLED`` (the eval harness needs sparse/hybrid
    constructible even when the flag is off), so a dense-only deployment
    must not pay for a local model load/download it will never use.
    """

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        self._model_name = model_name
        self._model: SparseTextEmbedding | None = None

    def _loaded_model(self) -> SparseTextEmbedding:
        if self._model is None:
            self._model = SparseTextEmbedding(model_name=self._model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        if not texts:
            return []
        return [
            SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())
            for embedding in self._loaded_model().embed(texts)
        ]

    def embed_query(self, text: str) -> SparseVector:
        embedding = next(iter(self._loaded_model().query_embed(text)))
        return SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())
