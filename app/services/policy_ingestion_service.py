"""Admin-triggered ingestion of a single uploaded policy document.

Wraps the same DI-provided pieces ``scripts/seed_db.py`` already uses
(``DocumentProcessor``, ``EmbeddingClient``, ``SparseEmbeddingClient``,
``VectorRepository``) so an admin upload is chunked/embedded/stored
exactly as the batch seed script would do it - no second ingestion code
path to drift out of sync.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from app.core.exceptions import FileTooLargeError, UnsupportedFileTypeError
from app.core.ingestion.document_processor import SUPPORTED_SUFFIXES, DocumentProcessor
from app.core.llm.embedding_client import EmbeddingClient
from app.core.llm.sparse_embedding_client import SparseEmbeddingClient
from app.repositories.vector_repository import INGESTION_STATUS_ACTIVE, VectorRepository
from app.schemas.policy import PolicyListResponse, PolicySummary, PolicyUploadResponse

logger = logging.getLogger(__name__)


class PolicyIngestionService:
    """Lists ingested policies and ingests newly-uploaded ones."""

    def __init__(
        self,
        document_processor: DocumentProcessor,
        embedding_client: EmbeddingClient,
        sparse_embedding_client: SparseEmbeddingClient,
        vector_repository: VectorRepository,
        policy_dir: Path,
        max_upload_size_mb: int,
    ) -> None:
        self._document_processor = document_processor
        self._embedding_client = embedding_client
        self._sparse_embedding_client = sparse_embedding_client
        self._vector_repository = vector_repository
        self._policy_dir = policy_dir
        self._max_upload_size_bytes = max_upload_size_mb * 1024 * 1024
        self._max_upload_size_mb = max_upload_size_mb

    def list_policies(self, status: str | None = INGESTION_STATUS_ACTIVE) -> PolicyListResponse:
        """``status`` defaults to ``active`` so the admin policy list keeps
        showing only live documents, unchanged from before the quarantine
        workflow existed; pass ``None`` for every status or a specific
        quarantine-state value."""
        counts: dict[str, int] = {}
        for record in self._vector_repository.scroll_all_chunks(ingestion_status=status):
            source = str(record.get("source") or "")
            if not source:
                continue
            counts[source] = counts.get(source, 0) + 1

        policies = [
            PolicySummary(source=source, chunk_count=count)
            for source, count in sorted(counts.items())
        ]
        return PolicyListResponse(policies=policies)

    def ingest(
        self,
        filename: str,
        content: bytes,
        ingestion_status: str = INGESTION_STATUS_ACTIVE,
    ) -> PolicyUploadResponse:
        """Chunk, embed, and store ``content`` under ``filename``.

        Re-uploading a filename that's already been ingested replaces it:
        the old chunks for that source (regardless of their prior
        ingestion status) are deleted before the new ones are upserted,
        rather than being rejected or left to accumulate alongside
        duplicates. ``ingestion_status`` defaults to ``active`` so every
        pre-existing caller (``scripts/seed_db.py``, tests) keeps its
        original always-searchable behavior unchanged; the quarantine
        upload path (``app.services.policy_ingestion_security_service``)
        passes ``quarantined`` explicitly.
        """
        self._validate(filename, content)

        existing_sources = {
            p.source for p in self.list_policies(status=None).policies
        }
        is_replacing = filename in existing_sources

        temp_dir = Path(tempfile.mkdtemp(prefix="policy_upload_"))
        try:
            # Docling derives the stored `source` from the file's own name
            # on disk, so the temp file must keep the original filename -
            # a random temp name would silently become the citation source.
            temp_path = temp_dir / filename
            temp_path.write_bytes(content)

            chunks = self._document_processor.process_document(str(temp_path))
            if not chunks:
                raise ValueError(f"No content could be extracted from '{filename}'")

            texts = [chunk.text for chunk in chunks]
            dense_embeddings = self._embedding_client.embed_texts(texts)
            sparse_embeddings = self._sparse_embedding_client.embed_documents(texts)

            self._policy_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(temp_path, self._policy_dir / filename)

            if is_replacing:
                self._vector_repository.delete_by_source(filename)
            self._vector_repository.upsert_chunks(
                chunks, dense_embeddings, sparse_embeddings, ingestion_status=ingestion_status
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        logger.info(
            "policy.ingested",
            extra={
                "source": filename,
                "chunk_count": len(chunks),
                "replaced": is_replacing,
                "ingestion_status": ingestion_status,
            },
        )
        return PolicyUploadResponse(
            source=filename,
            chunks_ingested=len(chunks),
            replaced=is_replacing,
            status=ingestion_status,
        )

    def _validate(self, filename: str, content: bytes) -> None:
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise UnsupportedFileTypeError(filename, SUPPORTED_SUFFIXES)
        if len(content) > self._max_upload_size_bytes:
            raise FileTooLargeError(filename, self._max_upload_size_mb)
