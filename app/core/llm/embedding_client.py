"""Adapter around the OpenAI embeddings API, with a cache-aside layer.

Same shape as ``chat_client.py``: wraps the OpenAI SDK behind an in-house
Protocol, and the one real external call (embedding creation) runs behind
the same retry treatment as ``chat_client``/``app.eval.ragas_adapter``.

Caching is cache-aside through ``QueryCacheService`` (``CacheTier.EMBEDDING``):
check cache per text, batch-fetch only the misses in one API call, write
them back. A cache read/write failure is logged and treated as a miss/no-op
rather than propagated — the cache is an optimization (embeddings are
deterministic and billed per token; caching just avoids re-paying for the
same text), not a correctness requirement, so a Redis outage shouldn't take
down embedding generation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import TYPE_CHECKING, Protocol, cast

from openai import OpenAI

from app.services.query_cache_service import CacheTier, QueryCacheService

if TYPE_CHECKING:
    from openai.types import CreateEmbeddingResponse

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 2.0


class EmbeddingClient(Protocol):
    """Contract for an embedding provider."""

    def embed_texts(self, texts: list[str], *, model: str | None = None) -> list[list[float]]: ...


class OpenAIEmbeddingClient:
    """:class:`EmbeddingClient` backed by the OpenAI SDK, with caching."""

    def __init__(self, client: OpenAI, cache: QueryCacheService, default_model: str) -> None:
        self._client = client
        self._cache = cache
        self._default_model = default_model

    def embed_texts(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        """Embed each text, in the same order, one vector per input.

        Duplicate texts within ``texts`` are only sent to the API once
        (embeddings are billed per token — no reason to pay twice in the
        same call). Raises if the response somehow leaves a gap, rather
        than silently returning fewer vectors than inputs — a caller
        zipping the result against ``texts`` must never see a length
        mismatch.
        """
        if not texts:
            return []
        resolved_model = model or self._default_model

        results: list[list[float] | None] = [None] * len(texts)
        miss_indices_by_text: dict[str, list[int]] = {}

        for i, text in enumerate(texts):
            cached = self._get_cached(text, resolved_model)
            if cached is not None:
                results[i] = cached
            else:
                miss_indices_by_text.setdefault(text, []).append(i)

        miss_texts = list(miss_indices_by_text.keys())
        if miss_texts:
            response = self._create_with_retries(resolved_model, miss_texts)
            for item in response.data:
                text = miss_texts[item.index]
                vector = item.embedding
                self._set_cached(text, resolved_model, vector)
                for original_idx in miss_indices_by_text[text]:
                    results[original_idx] = vector

        missing = [i for i, vector in enumerate(results) if vector is None]
        if missing:
            raise RuntimeError(f"embedding response missing vectors for indices {missing}")

        return cast(list[list[float]], results)

    def _create_with_retries(self, model: str, texts: list[str]) -> CreateEmbeddingResponse:
        last_error: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                return self._client.embeddings.create(input=texts, model=model)
            except Exception as exc:  # noqa: BLE001 - retry any transient API failure
                last_error = exc
                logger.warning(
                    "llm.embedding_retry",
                    extra={
                        "model": model,
                        "attempt": attempt,
                        "max_attempts": _MAX_ATTEMPTS,
                        "error": str(exc),
                    },
                )
                if attempt < _MAX_ATTEMPTS:
                    time.sleep(_RETRY_BACKOFF_SECONDS * attempt)

        assert last_error is not None  # loop always sets it before exhausting attempts
        raise last_error

    def _get_cached(self, text: str, model: str) -> list[float] | None:
        try:
            raw = self._cache.get(CacheTier.EMBEDDING, self._cache_key(text, model))
        except Exception as exc:  # noqa: BLE001 - cache is an optimization, not a correctness requirement
            logger.warning("llm.embedding_cache_read_failed", extra={"error": str(exc)})
            return None
        return json.loads(raw) if raw is not None else None

    def _set_cached(self, text: str, model: str, vector: list[float]) -> None:
        try:
            self._cache.set(CacheTier.EMBEDDING, self._cache_key(text, model), json.dumps(vector))
        except Exception as exc:  # noqa: BLE001 - cache is an optimization, not a correctness requirement
            logger.warning("llm.embedding_cache_write_failed", extra={"error": str(exc)})

    def _cache_key(self, text: str, model: str) -> str:
        # Keyed on model too: the same text embedded by two different
        # models produces different (and differently-shaped) vectors —
        # keying on text alone would serve a stale/wrong-dimension vector
        # after a model change. Hashed rather than raw text: embedding
        # inputs can be long chunks, and raw text as a Redis key risks
        # hitting practical key-size limits.
        digest = hashlib.sha256(f"{model}:{text}".encode()).hexdigest()
        return digest
