from __future__ import annotations

from types import SimpleNamespace

import pytest
from qdrant_client.models import CreateAliasOperation, DeleteAliasOperation

import scripts.migrate_qdrant_hybrid as migrate_qdrant_hybrid
from scripts.migrate_qdrant_hybrid import (
    _LIVE_ALIAS_SUFFIX,
    _cutover_alias,
    _require_complete_corpus,
)
from scripts.seed_db import SeedReport


class _FakeQdrantClient:
    """Enough of the QdrantClient surface for ``_cutover_alias`` - real
    alias/collection state kept in-memory so tests can assert on what
    actually ends up occupying each name, not just on call shape."""

    def __init__(
        self,
        aliases: dict[str, str] | None = None,
        bare_collections: set[str] | None = None,
    ) -> None:
        self._aliases = dict(aliases or {})
        self._bare_collections = set(bare_collections or set())
        self.update_calls: list[list[object]] = []

    def get_aliases(self):
        return SimpleNamespace(
            aliases=[
                SimpleNamespace(alias_name=name, collection_name=target)
                for name, target in self._aliases.items()
            ]
        )

    def collection_exists(self, name: str) -> bool:
        return name in self._bare_collections

    def update_collection_aliases(self, change_aliases_operations: list[object]) -> None:
        self.update_calls.append(change_aliases_operations)
        for op in change_aliases_operations:
            if isinstance(op, DeleteAliasOperation):
                self._aliases.pop(op.delete_alias.alias_name, None)
            elif isinstance(op, CreateAliasOperation):
                self._aliases[op.create_alias.alias_name] = op.create_alias.collection_name


def test_cutover_alias_swaps_an_existing_alias_in_a_single_atomic_call() -> None:
    """Once QDRANT_COLLECTION already names an alias, delete+create must
    be one update_collection_aliases() call - never two separate calls
    with a gap in between where the name resolves to nothing."""
    client = _FakeQdrantClient(aliases={"policy_documents": "policy_documents_v1"})

    _cutover_alias(client, "policy_documents", "policy_documents_v2")

    assert len(client.update_calls) == 1
    ops = client.update_calls[0]
    assert len(ops) == 2
    assert isinstance(ops[0], DeleteAliasOperation)
    assert isinstance(ops[1], CreateAliasOperation)
    assert client._aliases["policy_documents"] == "policy_documents_v2"


def test_cutover_alias_creates_directly_on_a_fresh_install() -> None:
    client = _FakeQdrantClient()

    _cutover_alias(client, "policy_documents", "policy_documents_v2")

    assert len(client.update_calls) == 1
    assert client._aliases["policy_documents"] == "policy_documents_v2"


def test_cutover_alias_uses_a_distinct_live_alias_for_a_bare_original_collection() -> None:
    """The very first migration can't reclaim the original bare
    collection's exact name without deleting it first (Qdrant forbids an
    alias sharing a name with a collection) - it must use a new alias
    name instead, and must never delete or touch the original."""
    client = _FakeQdrantClient(bare_collections={"policy_documents"})

    _cutover_alias(client, "policy_documents", "policy_documents_v2")

    live_alias = f"policy_documents{_LIVE_ALIAS_SUFFIX}"
    assert len(client.update_calls) == 1
    assert client._aliases[live_alias] == "policy_documents_v2"
    assert "policy_documents" not in client._aliases
    assert client.collection_exists("policy_documents")  # original untouched, never deleted


def test_cutover_alias_raises_if_the_live_alias_name_is_already_taken() -> None:
    client = _FakeQdrantClient(
        bare_collections={"policy_documents"},
        aliases={"policy_documents_live": "some_other_collection"},
    )

    with pytest.raises(RuntimeError, match="policy_documents_live"):
        _cutover_alias(client, "policy_documents", "policy_documents_v2")

    assert client.update_calls == []


def test_require_complete_corpus_accepts_a_fully_successful_report() -> None:
    report = SeedReport(
        discovered_files=2,
        succeeded_files=2,
        failed_files=[],
        chunks_written=5,
        sources=frozenset({"a.pdf", "b.pdf"}),
    )

    _require_complete_corpus(report)  # must not raise


def test_require_complete_corpus_rejects_any_failed_file() -> None:
    report = SeedReport(
        discovered_files=2,
        succeeded_files=1,
        failed_files=["b.pdf"],
        chunks_written=3,
        sources=frozenset({"a.pdf"}),
    )

    with pytest.raises(RuntimeError, match="b.pdf"):
        _require_complete_corpus(report)


def test_require_complete_corpus_rejects_a_partially_successful_report() -> None:
    """64 succeeded, 1 failed silently (succeeded_files short of
    discovered_files without necessarily populating failed_files) must
    still block cutover - the two checks are independent."""
    report = SeedReport(
        discovered_files=65, succeeded_files=64, failed_files=[], chunks_written=500
    )

    with pytest.raises(RuntimeError, match="64/65"):
        _require_complete_corpus(report)


def test_require_complete_corpus_rejects_an_empty_corpus() -> None:
    report = SeedReport(discovered_files=0, succeeded_files=0)

    with pytest.raises(RuntimeError, match="0 files"):
        _require_complete_corpus(report)


def test_blue_green_migrate_invalidates_caches_after_a_successful_cutover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: an alias cutover used to leave RAGService's caches
    pointed at the pre-migration corpus - stale answers/semantic pointers
    would keep being served after the alias already resolved to the newly
    (re)ingested collection. blue_green_migrate() must invalidate the
    corpus caches (see app.services.corpus_cache_invalidation_service) as
    its very last step, after the cutover has actually succeeded."""
    client = _FakeQdrantClient()
    client.count = lambda name: SimpleNamespace(count=3)  # type: ignore[method-assign]

    class _FakeNewRepository:
        def scroll_all_chunks(self):
            return [{"source": "a.pdf"}]

    report = SeedReport(
        discovered_files=1,
        succeeded_files=1,
        failed_files=[],
        chunks_written=3,
        sources=frozenset({"a.pdf"}),
    )

    invalidate_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        migrate_qdrant_hybrid,
        "get_settings",
        lambda: SimpleNamespace(qdrant=SimpleNamespace(qdrant_collection="policy_documents")),
    )
    monkeypatch.setattr(migrate_qdrant_hybrid, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(
        migrate_qdrant_hybrid,
        "QdrantVectorRepository",
        lambda client, collection_name: _FakeNewRepository(),
    )
    monkeypatch.setattr(migrate_qdrant_hybrid, "seed_docs", lambda vector_repository: report)
    monkeypatch.setattr(migrate_qdrant_hybrid, "get_rag_ops_repository", lambda: "repo")
    monkeypatch.setattr(migrate_qdrant_hybrid, "get_query_cache_service", lambda: "cache")
    monkeypatch.setattr(migrate_qdrant_hybrid, "get_semantic_query_cache", lambda: "semantic")
    monkeypatch.setattr(
        migrate_qdrant_hybrid,
        "invalidate_corpus_caches",
        lambda **kwargs: invalidate_calls.append(kwargs),
    )

    migrate_qdrant_hybrid.blue_green_migrate()

    assert client._aliases["policy_documents"] == "policy_documents_v2"
    assert len(invalidate_calls) == 1
    call = invalidate_calls[0]
    assert call["rag_ops_repository"] == "repo"
    assert call["query_cache"] == "cache"
    assert call["semantic_query_cache"] == "semantic"
    assert call["actor"] == "qdrant_migration"
    assert call["source"] == "policy_documents->policy_documents_v2"


def test_blue_green_migrate_does_not_invalidate_caches_on_an_incomplete_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial/failed ingest must abort before ever touching the alias or
    the caches - _require_complete_corpus's RuntimeError is the guard."""
    client = _FakeQdrantClient()
    report = SeedReport(discovered_files=2, succeeded_files=1, failed_files=["b.pdf"])

    invalidate_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        migrate_qdrant_hybrid,
        "get_settings",
        lambda: SimpleNamespace(qdrant=SimpleNamespace(qdrant_collection="policy_documents")),
    )
    monkeypatch.setattr(migrate_qdrant_hybrid, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(
        migrate_qdrant_hybrid, "QdrantVectorRepository", lambda client, collection_name: object()
    )
    monkeypatch.setattr(migrate_qdrant_hybrid, "seed_docs", lambda vector_repository: report)
    monkeypatch.setattr(
        migrate_qdrant_hybrid,
        "invalidate_corpus_caches",
        lambda **kwargs: invalidate_calls.append(kwargs),
    )

    with pytest.raises(RuntimeError, match="b.pdf"):
        migrate_qdrant_hybrid.blue_green_migrate()

    assert invalidate_calls == []
    assert client.update_calls == []
