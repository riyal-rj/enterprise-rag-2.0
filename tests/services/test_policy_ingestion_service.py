from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from app.core.exceptions import FileTooLargeError, UnsupportedFileTypeError
from app.core.ingestion.document_processor import DocumentChunk, DocumentProcessor
from app.core.llm.embedding_client import EmbeddingClient
from app.repositories.vector_repository import VectorRepository
from app.services.policy_ingestion_service import PolicyIngestionService


class _FakeDocumentProcessor:
    def __init__(self, chunks: list[DocumentChunk] | None = None) -> None:
        self._chunks = chunks if chunks is not None else [DocumentChunk(text="hello", source="x")]
        self.calls: list[str] = []

    def process_document(self, file_path: str) -> list[DocumentChunk]:
        self.calls.append(file_path)
        return self._chunks


class _FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.1, 0.2] for _ in texts]


class _FakeVectorRepository:
    def __init__(self, existing_records: list[dict[str, str | int | None]] | None = None) -> None:
        self._records = existing_records or []
        self.upsert_calls: list[tuple[list[DocumentChunk], list[list[float]]]] = []
        self.delete_calls: list[str] = []
        self.call_order: list[str] = []

    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        self.upsert_calls.append((chunks, embeddings))
        self.call_order.append("upsert")

    def search(self, query_embedding: list[float], top_k: int = 5):
        raise NotImplementedError

    def scroll_all_chunks(self, limit: int = 10_000) -> list[dict[str, str | int | None]]:
        return self._records

    def delete_by_source(self, source: str) -> None:
        self.delete_calls.append(source)
        self.call_order.append("delete")


def _service(
    tmp_path: Path,
    *,
    chunks: list[DocumentChunk] | None = None,
    existing_records: list[dict[str, str | int | None]] | None = None,
    max_upload_size_mb: int = 25,
) -> tuple[
    PolicyIngestionService, _FakeDocumentProcessor, _FakeEmbeddingClient, _FakeVectorRepository
]:
    processor = _FakeDocumentProcessor(chunks)
    embedder = _FakeEmbeddingClient()
    repository = _FakeVectorRepository(existing_records)
    service = PolicyIngestionService(
        document_processor=cast(DocumentProcessor, processor),
        embedding_client=cast(EmbeddingClient, embedder),
        vector_repository=cast(VectorRepository, repository),
        policy_dir=tmp_path / "policy",
        max_upload_size_mb=max_upload_size_mb,
    )
    return service, processor, embedder, repository


def test_ingest_fresh_document_upserts_without_deleting(tmp_path: Path) -> None:
    service, _processor, _embedder, repository = _service(tmp_path)

    response = service.ingest("refund-policy.pdf", b"pdf bytes")

    assert response.source == "refund-policy.pdf"
    assert response.chunks_ingested == 1
    assert response.replaced is False
    assert repository.delete_calls == []
    assert len(repository.upsert_calls) == 1


def test_ingest_existing_source_deletes_before_upserting(tmp_path: Path) -> None:
    service, _processor, _embedder, repository = _service(
        tmp_path,
        existing_records=[{"id": "1", "text": "old", "source": "refund-policy.pdf", "page_number": None}],
    )

    response = service.ingest("refund-policy.pdf", b"pdf bytes")

    assert response.replaced is True
    assert repository.call_order == ["delete", "upsert"]
    assert repository.delete_calls == ["refund-policy.pdf"]


def test_ingest_passes_the_original_filename_to_the_document_processor(tmp_path: Path) -> None:
    """Docling derives the stored `source` from the on-disk filename, so the
    temp file it's given must keep the original name, not a random temp one."""
    service, processor, _embedder, _repository = _service(tmp_path)

    service.ingest("refund-policy.pdf", b"pdf bytes")

    assert len(processor.calls) == 1
    assert Path(processor.calls[0]).name == "refund-policy.pdf"


def test_ingest_copies_the_file_into_the_policy_directory(tmp_path: Path) -> None:
    service, _processor, _embedder, _repository = _service(tmp_path)

    service.ingest("refund-policy.pdf", b"pdf bytes")

    assert (tmp_path / "policy" / "refund-policy.pdf").read_bytes() == b"pdf bytes"


def test_ingest_embeds_every_chunks_text(tmp_path: Path) -> None:
    chunks = [DocumentChunk(text="a", source="x"), DocumentChunk(text="b", source="x")]
    service, _processor, embedder, _repository = _service(tmp_path, chunks=chunks)

    service.ingest("multi-chunk.pdf", b"pdf bytes")

    assert embedder.calls == [["a", "b"]]


def test_ingest_rejects_unsupported_file_type(tmp_path: Path) -> None:
    service, processor, embedder, repository = _service(tmp_path)

    with pytest.raises(UnsupportedFileTypeError):
        service.ingest("notes.txt", b"plain text")

    assert processor.calls == []
    assert embedder.calls == []
    assert repository.upsert_calls == []


def test_ingest_rejects_oversized_content(tmp_path: Path) -> None:
    service, processor, _embedder, _repository = _service(tmp_path, max_upload_size_mb=1)

    with pytest.raises(FileTooLargeError):
        service.ingest("big.pdf", b"x" * (2 * 1024 * 1024))

    assert processor.calls == []


def test_ingest_raises_when_no_chunks_are_extracted(tmp_path: Path) -> None:
    service, _processor, embedder, repository = _service(tmp_path, chunks=[])

    with pytest.raises(ValueError, match="No content could be extracted"):
        service.ingest("empty.pdf", b"pdf bytes")

    assert embedder.calls == []
    assert repository.upsert_calls == []


def test_list_policies_groups_chunk_counts_by_source(tmp_path: Path) -> None:
    records: list[dict[str, str | int | None]] = [
        {"id": "1", "text": "a", "source": "a.pdf", "page_number": 1},
        {"id": "2", "text": "b", "source": "a.pdf", "page_number": 2},
        {"id": "3", "text": "c", "source": "b.pdf", "page_number": None},
    ]
    service, *_ = _service(tmp_path, existing_records=records)

    result = service.list_policies()

    assert [(p.source, p.chunk_count) for p in result.policies] == [
        ("a.pdf", 2),
        ("b.pdf", 1),
    ]
