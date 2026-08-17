"""Shared corpus-change cache invalidation.

Every corpus-mutating path - an admin's policy upload
(``AdminController.upload_policy``), a local/live ``seed_docs()`` run, and a
blue/green Qdrant alias cutover (``scripts/migrate_qdrant_hybrid.py``) -
must bump ``corpus_version`` and clear the caches that could otherwise keep
serving answers generated from the corpus version that just got replaced.
This is that one shared implementation, so a future corpus-mutation path
(document deletion, bulk replacement) doesn't have to remember to reinvent
it.

Deliberately targeted, not ``QueryCacheService.clear()``'s full wipe: a
corpus change only invalidates ``CacheTier.RAG_ANSWER`` answers and the
semantic-cache pointers into them. Clearing the embedding/SQL-gen/
SQL-result/intent tiers too would force an unrelated, unnecessary cold
start across the rest of the app for what is purely a policy-corpus change.

Belt-and-braces, not the only line of defense: ``RAGService`` also folds
``corpus_version`` into its cache namespace (see
``app.rag_services.rag_service``), so even if this invalidation is skipped
entirely - a Redis outage, a caller that forgets to call it - the bumped
``corpus_version`` alone makes every pre-bump cache entry unreachable on
the very next request, once that request's worker observes the new
version (immediately on the worker that bumped it; within one poll
interval on every other worker - see ``app.api.rag_ops_sync``).
"""

from __future__ import annotations

from app.repositories.rag_ops_repository import RagOpsRepository
from app.repositories.semantic_cache_repository import SemanticQueryCache
from app.services.query_cache_service import CacheTier, QueryCacheService


def invalidate_corpus_caches(
    *,
    rag_ops_repository: RagOpsRepository,
    query_cache: QueryCacheService,
    semantic_query_cache: SemanticQueryCache | None,
    actor: str,
    source: str,
) -> None:
    """Bump ``corpus_version`` and clear the RAG-answer-relevant caches.
    Call this after a corpus-mutating change has actually succeeded."""
    rag_ops_repository.bump_corpus_version(actor=actor, source=source)
    query_cache.clear_tiers({CacheTier.RAG_ANSWER})
    if semantic_query_cache is not None:
        semantic_query_cache.clear()
    rag_ops_repository.record_cache_invalidation()
