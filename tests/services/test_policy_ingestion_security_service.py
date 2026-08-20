"""Fast, fake-backed tests for the quarantine -> scan -> approve/reject
workflow - no real DB/Qdrant/model. Real Postgres/Qdrant integration is
covered separately (test_document_security_repository.py,
test_vector_repository_integration.py); this file is about the
orchestration logic in PolicyIngestionSecurityService itself.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import pytest
from qdrant_client.models import SparseVector

from app.core.ingestion.document_processor import DocumentChunk, DocumentProcessor
from app.core.llm.embedding_client import EmbeddingClient
from app.core.llm.sparse_embedding_client import SparseEmbeddingClient
from app.guardrails.contracts import GuardrailCategory, ScanFinding
from app.guardrails.ingestion_security import IngestionSecurityScanner
from app.guardrails.policy import GuardrailPolicy
from app.rag_services.rag_runtime_config import RagRuntimeConfig, RagRuntimeConfigStore
from app.repositories.document_security_repository import (
    DocumentSecurityRepository,
    DocumentSecurityState,
)
from app.repositories.vector_repository import VectorRepository
from app.services.policy_ingestion_security_service import (
    DocumentNotApprovableError,
    PolicyIngestionSecurityService,
)
from app.services.policy_ingestion_service import PolicyIngestionService
from tests.guardrails.fakes import FakeTextScanner


class _FakeDocumentProcessor:
    def process_document(self, file_path: str) -> list[DocumentChunk]:
        return [DocumentChunk(text="policy content", source="doc.pdf")]


class _FakeEmbeddingClient:
    def embed_texts(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


class _FakeSparseEmbeddingClient:
    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [SparseVector(indices=[1], values=[1.0]) for _ in texts]

    def embed_query(self, text: str) -> SparseVector:
        raise NotImplementedError


class _FakeVectorRepository:
    def __init__(self, quarantined_texts: list[str] | None = None) -> None:
        self._quarantined_texts = quarantined_texts or ["policy content"]
        self.upsert_calls: list[str] = []
        self.delete_calls: list[str] = []
        self.set_status_calls: list[tuple[str, str]] = []

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        dense_embeddings: list[list[float]],
        sparse_embeddings: list[SparseVector],
        ingestion_status: str = "active",
    ) -> None:
        self.upsert_calls.append(ingestion_status)

    def search_dense(self, query_embedding: list[float], top_k: int = 5):
        raise NotImplementedError

    def search_sparse(self, query_sparse: SparseVector, top_k: int = 5):
        raise NotImplementedError

    def search_hybrid(self, *args: object, **kwargs: object):
        raise NotImplementedError

    def scroll_all_chunks(
        self, limit: int = 10_000, ingestion_status: str | None = "active"
    ) -> list[dict[str, str | int | None]]:
        if ingestion_status == "quarantined":
            return [
                {"id": str(i), "text": text, "source": "doc.pdf", "page_number": None}
                for i, text in enumerate(self._quarantined_texts)
            ]
        return []

    def set_ingestion_status(self, source: str, ingestion_status: str) -> None:
        self.set_status_calls.append((source, ingestion_status))

    def delete_by_source(self, source: str) -> None:
        self.delete_calls.append(source)


class _FakeDocumentSecurityRepository:
    def __init__(self) -> None:
        self._states: dict[int, DocumentSecurityState] = {}
        self._next_id = 1
        self.record_scan_calls: list[tuple[int, str]] = []

    def create_pending(
        self, *, source: str, uploaded_by: str, chunk_count: int
    ) -> DocumentSecurityState:
        state = DocumentSecurityState(
            id=self._next_id,
            source=source,
            status="pending_scan",
            uploaded_by=uploaded_by,
            uploaded_at=datetime.now(UTC),
            scan_decision=None,
            scanned_at=None,
            approved_by=None,
            approved_at=None,
            rejected_reason=None,
            chunk_count=chunk_count,
        )
        self._states[state.id] = state
        self._next_id += 1
        return state

    def record_scan_result(self, *, id: int, status: str, scan_decision: dict) -> DocumentSecurityState:
        self.record_scan_calls.append((id, status))
        state = self._states[id]
        updated = DocumentSecurityState(
            id=state.id,
            source=state.source,
            status=status,
            uploaded_by=state.uploaded_by,
            uploaded_at=state.uploaded_at,
            scan_decision=scan_decision,
            scanned_at=datetime.now(UTC),
            approved_by=state.approved_by,
            approved_at=state.approved_at,
            rejected_reason=state.rejected_reason,
            chunk_count=state.chunk_count,
        )
        self._states[id] = updated
        return updated

    def approve(self, *, id: int, actor: str) -> DocumentSecurityState:
        state = self._states[id]
        updated = DocumentSecurityState(
            id=state.id,
            source=state.source,
            status="approved",
            uploaded_by=state.uploaded_by,
            uploaded_at=state.uploaded_at,
            scan_decision=state.scan_decision,
            scanned_at=state.scanned_at,
            approved_by=actor,
            approved_at=datetime.now(UTC),
            rejected_reason=state.rejected_reason,
            chunk_count=state.chunk_count,
        )
        self._states[id] = updated
        return updated

    def mark_active(self, *, id: int) -> DocumentSecurityState:
        state = self._states[id]
        updated_dict = state.__dict__ | {"status": "active"}
        updated = DocumentSecurityState(**updated_dict)
        self._states[id] = updated
        return updated

    def reject(self, *, id: int, actor: str, reason: str) -> DocumentSecurityState:
        state = self._states[id]
        updated_dict = state.__dict__ | {"status": "rejected", "rejected_reason": reason}
        updated = DocumentSecurityState(**updated_dict)
        self._states[id] = updated
        return updated

    def get_latest_for_source(self, source: str) -> DocumentSecurityState | None:
        matches = [s for s in self._states.values() if s.source == source]
        return max(matches, key=lambda s: s.id) if matches else None

    def list_pending_approval(self, limit: int) -> list[DocumentSecurityState]:
        return [s for s in self._states.values() if s.status == "scan_passed"][:limit]


def _service(
    *,
    vector_repository: _FakeVectorRepository | None = None,
    findings: tuple[ScanFinding, ...] = (),
    guardrail_mode: str = "enforce",
) -> tuple[PolicyIngestionSecurityService, _FakeVectorRepository, _FakeDocumentSecurityRepository]:
    from pathlib import Path

    vector_repo = vector_repository or _FakeVectorRepository()
    ingestion = PolicyIngestionService(
        document_processor=cast(DocumentProcessor, _FakeDocumentProcessor()),
        embedding_client=cast(EmbeddingClient, _FakeEmbeddingClient()),
        sparse_embedding_client=cast(SparseEmbeddingClient, _FakeSparseEmbeddingClient()),
        vector_repository=cast(VectorRepository, vector_repo),
        policy_dir=Path("/tmp/policy_test"),
        max_upload_size_mb=25,
    )
    security_repository = _FakeDocumentSecurityRepository()
    scanner = IngestionSecurityScanner(
        deterministic_scanners=(),
        ml_scanner=FakeTextScanner(findings),
        policy=GuardrailPolicy(),
    )
    config_store = RagRuntimeConfigStore(_config(guardrail_mode=guardrail_mode))
    service = PolicyIngestionSecurityService(
        ingestion=ingestion,
        security_repository=cast(DocumentSecurityRepository, security_repository),
        scanner=scanner,
        vector_repository=cast(VectorRepository, vector_repo),
        config_store=config_store,
    )
    return service, vector_repo, security_repository


def _config(*, guardrail_mode: str = "enforce") -> RagRuntimeConfig:
    return RagRuntimeConfig(
        reranking_enabled=False,
        reranker_backend="local",
        reranker_rollout_percentage=100,
        emergency_disabled=False,
        semantic_cache_enabled=False,
        semantic_cache_threshold=0.95,
        corpus_version=1,
        hyde_enabled=False,
        hyde_rollout_percentage=0,
        crag_enabled=False,
        crag_rollout_percentage=0,
        crag_web_enabled=False,
        guardrail_mode=guardrail_mode,  # type: ignore[arg-type]
    )


def test_submit_uploads_as_quarantined_not_active() -> None:
    service, vector_repo, _security = _service()

    service.submit("doc.pdf", b"content", "admin")

    assert vector_repo.upsert_calls == ["quarantined"]


def test_clean_document_passes_scan() -> None:
    service, _vector_repo, _security = _service(findings=())

    response = service.submit("doc.pdf", b"content", "admin")

    assert response.status == "scan_passed"


def test_injection_flagged_document_fails_scan_and_is_deleted_from_qdrant() -> None:
    vector_repo = _FakeVectorRepository()
    service, _vector_repo, security = _service(
        vector_repository=vector_repo,
        findings=(ScanFinding(GuardrailCategory.PROMPT_INJECTION, 0.98, "fake"),),
    )

    response = service.submit("doc.pdf", b"content", "admin")

    assert response.status == "scan_failed"
    assert vector_repo.delete_calls == ["doc.pdf"]  # never left quarantined indefinitely
    state = security.get_latest_for_source("doc.pdf")
    assert state is not None
    assert state.status == "scan_failed"


def test_monitor_mode_lets_a_flagged_document_pass_the_scan() -> None:
    service, vector_repo, _security = _service(
        findings=(ScanFinding(GuardrailCategory.PROMPT_INJECTION, 0.98, "fake"),),
        guardrail_mode="monitor",
    )

    response = service.submit("doc.pdf", b"content", "admin")

    assert response.status == "scan_passed"
    assert vector_repo.delete_calls == []


def test_approve_requires_scan_passed_status() -> None:
    service, _vector_repo, _security = _service()

    with pytest.raises(DocumentNotApprovableError):
        service.approve("never-uploaded.pdf", "admin")


def test_approve_activates_in_qdrant_and_marks_the_row_active() -> None:
    service, vector_repo, security = _service(findings=())
    service.submit("doc.pdf", b"content", "admin")

    state = service.approve("doc.pdf", "security-admin")

    assert state.status == "active"
    assert vector_repo.set_status_calls == [("doc.pdf", "active")]


def test_approve_rejects_a_scan_failed_document() -> None:
    service, _vector_repo, _security = _service(
        findings=(ScanFinding(GuardrailCategory.SECRETS, 1.0, "fake"),)
    )
    service.submit("doc.pdf", b"content", "admin")

    with pytest.raises(DocumentNotApprovableError):
        service.approve("doc.pdf", "admin")


def test_reject_deletes_the_quarantined_chunks() -> None:
    service, vector_repo, security = _service(findings=())
    service.submit("doc.pdf", b"content", "admin")

    state = service.reject("doc.pdf", "admin", "policy violates internal guidelines")

    assert state.status == "rejected"
    assert state.rejected_reason == "policy violates internal guidelines"
    assert vector_repo.delete_calls == ["doc.pdf"]


def test_reject_an_already_rejected_document_raises() -> None:
    service, _vector_repo, _security = _service(findings=())
    service.submit("doc.pdf", b"content", "admin")
    service.reject("doc.pdf", "admin", "first rejection")

    with pytest.raises(DocumentNotApprovableError):
        service.reject("doc.pdf", "admin", "second attempt")
