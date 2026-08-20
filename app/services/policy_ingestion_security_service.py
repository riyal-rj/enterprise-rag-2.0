"""Quarantine -> scan -> approve -> active workflow around
``PolicyIngestionService``.

Wraps rather than rewrites: ``PolicyIngestionService.ingest()``'s existing
chunk/embed/store logic is untouched and still directly usable by anything
that wants immediate, unquarantined ingestion (``scripts/seed_db.py``,
which seeds a known-good demo corpus and has no admin to approve it). This
service adds the security gate in front of the one path that matters for
production: an admin uploading a new policy document through ``/admin/policies``.

State machine (mirrors ``document_security_state.status``'s CHECK
constraint): ``pending_scan`` (row created, chunks upserted into Qdrant as
``quarantined``) -> ``scan_passed``/``scan_failed`` (synchronous scan, see
``app.guardrails.ingestion_security``) -> an admin approval moves
``scan_passed`` -> ``approved`` -> ``active`` (Qdrant payload flipped via a
targeted ``set_ingestion_status`` call, no re-embedding). ``scan_failed``
and explicitly-rejected documents have their Qdrant chunks deleted outright
rather than left quarantined indefinitely.
"""

from __future__ import annotations

import logging

from app.guardrails.contracts import GuardrailAction, GuardrailDecision
from app.guardrails.ingestion_security import IngestionSecurityScanner
from app.rag_services.rag_runtime_config import RagRuntimeConfigStore
from app.repositories.document_security_repository import (
    DocumentSecurityRepository,
    DocumentSecurityState,
)
from app.repositories.vector_repository import INGESTION_STATUS_QUARANTINED, VectorRepository
from app.schemas.policy import PolicyListResponse, PolicyUploadResponse
from app.services.policy_ingestion_service import PolicyIngestionService

logger = logging.getLogger(__name__)


class DocumentNotApprovableError(ValueError):
    """Raised when ``approve``/``reject`` is called for a source with no
    row in ``scan_passed`` state to act on (already decided, never
    uploaded, or still mid-scan)."""


class PolicyIngestionSecurityService:
    def __init__(
        self,
        *,
        ingestion: PolicyIngestionService,
        security_repository: DocumentSecurityRepository,
        scanner: IngestionSecurityScanner,
        vector_repository: VectorRepository,
        config_store: RagRuntimeConfigStore,
    ) -> None:
        self._ingestion = ingestion
        self._security_repository = security_repository
        self._scanner = scanner
        self._vector_repository = vector_repository
        self._config_store = config_store

    def list_policies(self, status: str | None = "active") -> PolicyListResponse:
        """Passthrough to the wrapped service - kept here too so
        ``AdminController`` needs only this one dependency for every
        policy-admin operation."""
        return self._ingestion.list_policies(status=status)

    def submit(self, filename: str, content: bytes, actor: str) -> PolicyUploadResponse:
        response = self._ingestion.ingest(
            filename, content, ingestion_status=INGESTION_STATUS_QUARANTINED
        )
        state = self._security_repository.create_pending(
            source=filename, uploaded_by=actor, chunk_count=response.chunks_ingested
        )

        chunk_texts = self._quarantined_chunk_texts(filename)
        mode = self._config_store.current.guardrail_mode
        decision = self._scanner.scan_document(chunk_texts, mode=mode, actor=actor)

        if decision.action is GuardrailAction.BLOCK:
            self._security_repository.record_scan_result(
                id=state.id, status="scan_failed", scan_decision=_summarize(decision)
            )
            # Never left quarantined indefinitely - a scan-failed document
            # has no path to ever become searchable, so its (unsearchable)
            # chunks are removed outright rather than lingering forever.
            self._vector_repository.delete_by_source(filename)
            logger.warning(
                "policy.ingestion_scan_failed",
                extra={"source": filename, "finding_count": len(decision.findings)},
            )
            return response.model_copy(update={"status": "scan_failed"})

        self._security_repository.record_scan_result(
            id=state.id, status="scan_passed", scan_decision=_summarize(decision)
        )
        logger.info(
            "policy.ingestion_scan_passed",
            extra={"source": filename, "finding_count": len(decision.findings)},
        )
        return response.model_copy(update={"status": "scan_passed"})

    def approve(self, source: str, actor: str) -> DocumentSecurityState:
        state = self._security_repository.get_latest_for_source(source)
        if state is None or state.status != "scan_passed":
            raise DocumentNotApprovableError(
                f"'{source}' has no scan_passed upload awaiting approval"
            )
        self._security_repository.approve(id=state.id, actor=actor)
        self._vector_repository.set_ingestion_status(source, "active")
        activated = self._security_repository.mark_active(id=state.id)
        logger.info("policy.ingestion_approved", extra={"source": source, "actor": actor})
        return activated

    def reject(self, source: str, actor: str, reason: str) -> DocumentSecurityState:
        state = self._security_repository.get_latest_for_source(source)
        if state is None or state.status not in ("scan_passed", "scan_failed"):
            raise DocumentNotApprovableError(f"'{source}' has no upload awaiting a decision")
        self._vector_repository.delete_by_source(source)
        rejected = self._security_repository.reject(id=state.id, actor=actor, reason=reason)
        logger.info(
            "policy.ingestion_rejected", extra={"source": source, "actor": actor, "reason": reason}
        )
        return rejected

    def list_pending(self, limit: int = 50) -> list[DocumentSecurityState]:
        return self._security_repository.list_pending_approval(limit)

    def _quarantined_chunk_texts(self, source: str) -> list[str]:
        """Reads back the just-upserted chunk text for scanning, rather
        than re-processing the document - avoids a second Docling parse of
        the same file and keeps this service's only contract with
        ``PolicyIngestionService`` its existing public ``ingest()`` method."""
        records = self._vector_repository.scroll_all_chunks(
            ingestion_status=INGESTION_STATUS_QUARANTINED
        )
        return [str(r["text"]) for r in records if r.get("source") == source]


def _summarize(decision: GuardrailDecision) -> dict[str, object]:
    """Sanitized finding summary for ``document_security_state.scan_decision``
    - category/score/detector only, never raw scanned text, same discipline
    every other guardrail audit surface applies."""
    return {
        "action": decision.action.value,
        "findings": [
            {"category": f.category.value, "score": f.score, "detector": f.detector}
            for f in decision.findings
        ],
    }
