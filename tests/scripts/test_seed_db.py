from __future__ import annotations

from pathlib import Path
from typing import cast

from qdrant_client.models import SparseVector

from app.core.ingestion.document_processor import DocumentChunk
from app.repositories.vector_repository import VectorRepository
from scripts import seed_db


class _FakeProcessor:
    def __init__(
        self,
        chunks_by_file: dict[str, list[DocumentChunk]],
        fail_on: frozenset[str] = frozenset(),
    ) -> None:
        self._chunks_by_file = chunks_by_file
        self._fail_on = fail_on

    def process_document(self, file_path: str) -> list[DocumentChunk]:
        name = Path(file_path).name
        if name in self._fail_on:
            raise RuntimeError(f"boom: {name}")
        return self._chunks_by_file[name]


class _FakeEmbedder:
    def embed_texts(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        return [[0.1] for _ in texts]


class _FakeSparseEmbedder:
    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [SparseVector(indices=[1], values=[1.0]) for _ in texts]

    def embed_query(self, text: str) -> SparseVector:
        raise NotImplementedError


class _FakeRepo:
    def __init__(self) -> None:
        self.upserted: list[list[DocumentChunk]] = []

    def upsert_chunks(self, chunks, dense_embeddings, sparse_embeddings) -> None:
        self.upserted.append(chunks)

    def search_dense(self, *args, **kwargs):
        raise NotImplementedError

    def search_sparse(self, *args, **kwargs):
        raise NotImplementedError

    def search_hybrid(self, *args, **kwargs):
        raise NotImplementedError

    def scroll_all_chunks(self, limit: int = 10_000):
        raise NotImplementedError

    def delete_by_source(self, source: str) -> None:
        raise NotImplementedError


class _FakeRagOpsRepository:
    def __init__(self) -> None:
        self.bump_calls: list[dict[str, object]] = []
        self.invalidation_calls = 0

    def bump_corpus_version(self, *, actor: str, source: str):
        self.bump_calls.append({"actor": actor, "source": source})

    def record_cache_invalidation(self):
        self.invalidation_calls += 1


class _FakeQueryCacheService:
    def __init__(self) -> None:
        self.clear_calls = 0
        self.clear_tiers_calls: list[set] = []

    def clear_tiers(self, tiers: set) -> dict[str, int]:
        self.clear_calls += 1
        self.clear_tiers_calls.append(tiers)
        return {}


class _FakeLiveSemanticQueryCache:
    def __init__(self) -> None:
        self.clear_calls = 0

    def clear(self) -> int:
        self.clear_calls += 1
        return 0


def _patch_di(monkeypatch, processor: _FakeProcessor) -> None:
    monkeypatch.setattr(seed_db, "get_document_processor", lambda: processor)
    monkeypatch.setattr(seed_db, "get_embedding_client", lambda: _FakeEmbedder())
    monkeypatch.setattr(seed_db, "get_sparse_embedding_client", lambda: _FakeSparseEmbedder())


def test_seed_docs_reports_full_success(monkeypatch, tmp_path) -> None:
    (tmp_path / "a.pdf").write_bytes(b"x")
    (tmp_path / "b.docx").write_bytes(b"x")
    processor = _FakeProcessor(
        {
            "a.pdf": [DocumentChunk(text="a1", source="a.pdf")],
            "b.docx": [
                DocumentChunk(text="b1", source="b.docx"),
                DocumentChunk(text="b2", source="b.docx"),
            ],
        }
    )
    _patch_di(monkeypatch, processor)
    repo = _FakeRepo()

    report = seed_db.seed_docs(policy_dir=tmp_path, vector_repository=cast(VectorRepository, repo))

    assert report.discovered_files == 2
    assert report.succeeded_files == 2
    assert report.failed_files == []
    assert report.chunks_written == 3
    assert report.sources == frozenset({"a.pdf", "b.docx"})


def test_seed_docs_reports_failures_without_aborting_the_batch(monkeypatch, tmp_path) -> None:
    """A corrupt/unsupported file must be recorded as a failure, not
    silently dropped - the migration's cutover gate depends on this."""
    (tmp_path / "a.pdf").write_bytes(b"x")
    (tmp_path / "bad.pdf").write_bytes(b"x")
    processor = _FakeProcessor(
        {"a.pdf": [DocumentChunk(text="a1", source="a.pdf")]}, fail_on=frozenset({"bad.pdf"})
    )
    _patch_di(monkeypatch, processor)
    repo = _FakeRepo()

    report = seed_db.seed_docs(policy_dir=tmp_path, vector_repository=cast(VectorRepository, repo))

    assert report.discovered_files == 2
    assert report.succeeded_files == 1
    assert report.failed_files == ["bad.pdf"]
    assert report.chunks_written == 1
    assert report.sources == frozenset({"a.pdf"})


def test_seed_docs_treats_zero_chunks_extracted_as_a_failure(monkeypatch, tmp_path) -> None:
    (tmp_path / "empty.pdf").write_bytes(b"x")
    processor = _FakeProcessor({"empty.pdf": []})
    _patch_di(monkeypatch, processor)
    repo = _FakeRepo()

    report = seed_db.seed_docs(policy_dir=tmp_path, vector_repository=cast(VectorRepository, repo))

    assert report.failed_files == ["empty.pdf"]
    assert report.succeeded_files == 0
    assert report.chunks_written == 0


def test_seed_docs_reports_zero_discovered_files_on_an_empty_directory(
    monkeypatch, tmp_path
) -> None:
    processor = _FakeProcessor({})
    _patch_di(monkeypatch, processor)
    repo = _FakeRepo()

    report = seed_db.seed_docs(policy_dir=tmp_path, vector_repository=cast(VectorRepository, repo))

    assert report.discovered_files == 0
    assert report.succeeded_files == 0
    assert report.failed_files == []


def test_seed_docs_against_the_live_collection_bumps_corpus_version_and_clears_caches(
    monkeypatch, tmp_path
) -> None:
    (tmp_path / "a.pdf").write_bytes(b"x")
    processor = _FakeProcessor({"a.pdf": [DocumentChunk(text="a1", source="a.pdf")]})
    _patch_di(monkeypatch, processor)
    live_repo = _FakeRepo()
    rag_ops_repository = _FakeRagOpsRepository()
    query_cache = _FakeQueryCacheService()
    semantic_cache = _FakeLiveSemanticQueryCache()
    monkeypatch.setattr(seed_db, "get_vector_repository", lambda: live_repo)
    monkeypatch.setattr(seed_db, "get_rag_ops_repository", lambda: rag_ops_repository)
    monkeypatch.setattr(seed_db, "get_query_cache_service", lambda: query_cache)
    monkeypatch.setattr(seed_db, "get_semantic_query_cache", lambda: semantic_cache)

    report = seed_db.seed_docs(policy_dir=tmp_path)

    assert report.succeeded_files == 1
    assert len(rag_ops_repository.bump_calls) == 1
    assert query_cache.clear_calls == 1
    assert semantic_cache.clear_calls == 1
    assert rag_ops_repository.invalidation_calls == 1


def test_seed_docs_against_the_live_collection_skips_invalidation_when_nothing_succeeded(
    monkeypatch, tmp_path
) -> None:
    (tmp_path / "empty.pdf").write_bytes(b"x")
    processor = _FakeProcessor({"empty.pdf": []})
    _patch_di(monkeypatch, processor)
    live_repo = _FakeRepo()
    rag_ops_repository = _FakeRagOpsRepository()
    query_cache = _FakeQueryCacheService()
    semantic_cache = _FakeLiveSemanticQueryCache()
    monkeypatch.setattr(seed_db, "get_vector_repository", lambda: live_repo)
    monkeypatch.setattr(seed_db, "get_rag_ops_repository", lambda: rag_ops_repository)
    monkeypatch.setattr(seed_db, "get_query_cache_service", lambda: query_cache)
    monkeypatch.setattr(seed_db, "get_semantic_query_cache", lambda: semantic_cache)

    report = seed_db.seed_docs(policy_dir=tmp_path)

    assert report.succeeded_files == 0
    assert rag_ops_repository.bump_calls == []
    assert query_cache.clear_calls == 0
    assert semantic_cache.clear_calls == 0


def test_seed_docs_with_an_explicit_repository_never_bumps_corpus_version(
    monkeypatch, tmp_path
) -> None:
    """Regression: scripts/migrate_qdrant_hybrid.py stages a new physical
    collection under an explicit vector_repository before any alias
    cutover - that content isn't visible to the running app yet, so
    corpus_version/caches must be untouched even though ingestion succeeds."""
    (tmp_path / "a.pdf").write_bytes(b"x")
    processor = _FakeProcessor({"a.pdf": [DocumentChunk(text="a1", source="a.pdf")]})
    _patch_di(monkeypatch, processor)
    staging_repo = _FakeRepo()
    rag_ops_repository = _FakeRagOpsRepository()
    monkeypatch.setattr(seed_db, "get_rag_ops_repository", lambda: rag_ops_repository)

    report = seed_db.seed_docs(
        policy_dir=tmp_path, vector_repository=cast(VectorRepository, staging_repo)
    )

    assert report.succeeded_files == 1
    assert rag_ops_repository.bump_calls == []
