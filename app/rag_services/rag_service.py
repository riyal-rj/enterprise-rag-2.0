"""Basic RAG answer generation: embed, dense-search, generate, cache.

The full pipeline this is meant to grow into - CRAG relevance grading,
self-reflective answer refinement, HyDE query expansion, reranking,
hybrid/sparse search, SQL intent routing, prompt-injection spotlighting -
layers on top of this. This is deliberately the minimal "retrieve then
generate" core those strategies plug into later, not a placeholder for
any one of them: no query expansion, no relevance grading, no answer
refinement loop, and no prompt-injection defense yet.

Caching follows the same cache-aside shape as
``app.core.llm.embedding_client``: check the cache first, compute and
write back on a miss, treat a cache read/write failure as a miss/no-op
rather than propagate it - the cache is an optimization, not a
correctness requirement.
"""

from __future__ import annotations

import hashlib
import logging

from app.core.llm.chat_client import LLMClient
from app.core.llm.embedding_client import EmbeddingClient
from app.models.retrieved_chunk import RetrievedChunk
from app.repositories.vector_repository import VectorRepository
from app.schemas.chat import ChatResponse, ResponseMetadata, RetrievedChunkPreview
from app.services.query_cache_service import CacheTier, QueryCacheService

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the "
    "provided context. If the context doesn't contain the answer, say so "
    "instead of guessing. Cite sources by filename in brackets, e.g. "
    "[policy.pdf]."
)


class RAGService:
    """Retrieve-then-generate RAG answering, with response caching."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        vector_repository: VectorRepository,
        llm_client: LLMClient,
        cache: QueryCacheService,
    ) -> None:
        self._embedding_client = embedding_client
        self._vector_repository = vector_repository
        self._llm_client = llm_client
        self._cache = cache

    def answer(self, question: str, top_k: int = 5) -> ChatResponse:
        cache_key = self._cache_key(question, top_k)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached.model_copy(update={"cache_hit": True})

        query_embedding = self._embedding_client.embed_texts([question])[0]
        chunks = self._vector_repository.search(query_embedding, top_k=top_k)

        user_message = f"{self._build_context(chunks)}\n\nQuestion: {question}"
        llm_response = self._llm_client.generate(_SYSTEM_PROMPT, user_message)

        response = ChatResponse(
            answer=llm_response.text,
            sources=sorted({chunk.source for chunk in chunks}),
            confidence=0.7,
            metadata=ResponseMetadata(
                route="rag",
                retrieved_chunks=[
                    RetrievedChunkPreview(text=c.text, source=c.source, score=c.score)
                    for c in chunks
                ],
            ),
        )
        self._set_cached(cache_key, response)
        return response

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "No relevant context was found."
        return "\n\n".join(f"[{chunk.source}]\n{chunk.text}" for chunk in chunks)

    def _cache_key(self, question: str, top_k: int) -> str:
        digest = hashlib.sha256(f"{top_k}:{question}".encode()).hexdigest()
        return digest

    def _get_cached(self, key: str) -> ChatResponse | None:
        try:
            raw = self._cache.get(CacheTier.RAG_ANSWER, key)
        except Exception as exc:  # noqa: BLE001 - cache is an optimization, not a correctness requirement
            logger.warning("rag.cache_read_failed", extra={"error": str(exc)})
            return None
        return ChatResponse.model_validate_json(raw) if raw is not None else None

    def _set_cached(self, key: str, response: ChatResponse) -> None:
        try:
            self._cache.set(CacheTier.RAG_ANSWER, key, response.model_dump_json())
        except Exception as exc:  # noqa: BLE001 - cache is an optimization, not a correctness requirement
            logger.warning("rag.cache_write_failed", extra={"error": str(exc)})
