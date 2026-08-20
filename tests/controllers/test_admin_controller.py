from __future__ import annotations

from datetime import UTC

from app.controllers.admin_controller import AdminController
from app.core.config.cache import CacheSettings
from app.models.rag_ops import RagOpsAuditEntry, RagOpsConfig
from app.models.security_event import SecurityEvent
from app.repositories.document_security_repository import DocumentSecurityState
from app.schemas.policy import PolicyListResponse, PolicyUploadResponse
from app.services.query_cache_service import CacheTier, QueryCacheService


class _InMemoryCacheBackend:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self._store[key] = value

    def delete_prefix(self, prefix: str) -> int:
        keys = [k for k in self._store if k.startswith(prefix)]
        for k in keys:
            del self._store[k]
        return len(keys)


def _document_state(source: str, status: str) -> DocumentSecurityState:
    from datetime import datetime

    return DocumentSecurityState(
        id=1,
        source=source,
        status=status,
        uploaded_by="admin",
        uploaded_at=datetime.now(UTC),
        scan_decision=None,
        scanned_at=None,
        approved_by=None,
        approved_at=None,
        rejected_reason=None,
        chunk_count=3,
    )


class _FakePolicyIngestionSecurityService:
    """Mirrors PolicyIngestionSecurityService's public surface - submit/
    approve/reject/list_pending/list_policies - not the old, pre-quarantine
    PolicyIngestionService.ingest() shape."""

    def __init__(self) -> None:
        self.submit_calls: list[tuple[str, bytes, str]] = []
        self.approve_calls: list[tuple[str, str]] = []
        self.reject_calls: list[tuple[str, str, str]] = []

    def list_policies(self) -> PolicyListResponse:
        raise NotImplementedError

    def list_pending(self) -> list[DocumentSecurityState]:
        return []

    def submit(self, filename: str, content: bytes, actor: str) -> PolicyUploadResponse:
        self.submit_calls.append((filename, content, actor))
        return PolicyUploadResponse(
            source=filename, chunks_ingested=3, replaced=False, status="scan_passed"
        )

    def approve(self, source: str, actor: str) -> DocumentSecurityState:
        self.approve_calls.append((source, actor))
        return _document_state(source, "active")

    def reject(self, source: str, actor: str, reason: str) -> DocumentSecurityState:
        self.reject_calls.append((source, actor, reason))
        return _document_state(source, "rejected")


class _FakeSemanticQueryCache:
    def __init__(self) -> None:
        self.clear_calls = 0

    def clear(self) -> int:
        self.clear_calls += 1
        return 5


class _FakeRagOpsRepository:
    def __init__(self) -> None:
        self.bump_calls: list[dict[str, object]] = []
        self.invalidation_calls = 0

    def bump_corpus_version(self, *, actor: str, source: str) -> RagOpsConfig:
        self.bump_calls.append({"actor": actor, "source": source})
        return _config()

    def record_cache_invalidation(self) -> RagOpsConfig:
        self.invalidation_calls += 1
        return _config()

    def get_config(self) -> RagOpsConfig:
        raise NotImplementedError

    def update_config(self, **kwargs: object) -> RagOpsConfig:
        raise NotImplementedError

    def set_emergency_disabled(
        self, *, actor: str, disabled: bool, reason: str | None
    ) -> RagOpsConfig:
        raise NotImplementedError

    def list_audit(self, limit: int) -> list[RagOpsAuditEntry]:
        raise NotImplementedError


def _config() -> RagOpsConfig:
    from datetime import datetime

    return RagOpsConfig(
        reranking_enabled=False,
        reranker_backend="local",
        reranker_rollout_percentage=100,
        semantic_cache_enabled=False,
        semantic_cache_threshold=0.95,
        hyde_enabled=False,
        hyde_rollout_percentage=0,
        crag_enabled=False,
        crag_rollout_percentage=0,
        crag_web_enabled=False,
        emergency_disabled=False,
        emergency_disabled_reason=None,
        emergency_disabled_at=None,
        emergency_disabled_by=None,
        corpus_version=2,
        last_cache_invalidated_at=None,
        updated_at=datetime.now(UTC),
        updated_by=None,
    )


class _FakeSecurityEventsRepository:
    def __init__(self, events: list[SecurityEvent] | None = None) -> None:
        self._events = events or []
        self.list_recent_calls: list[int] = []

    def record(self, **kwargs: object) -> SecurityEvent:
        raise NotImplementedError

    def list_recent(self, limit: int) -> list[SecurityEvent]:
        self.list_recent_calls.append(limit)
        return self._events[:limit]


def _controller(
    *,
    rag_ops_repository: _FakeRagOpsRepository | None = None,
    security_events_repository: _FakeSecurityEventsRepository | None = None,
) -> tuple[
    AdminController, QueryCacheService, _FakeSemanticQueryCache, _FakePolicyIngestionSecurityService
]:
    query_cache = QueryCacheService(_InMemoryCacheBackend(), CacheSettings())
    semantic_cache = _FakeSemanticQueryCache()
    ingestion = _FakePolicyIngestionSecurityService()
    controller = AdminController(
        health_check_service=None,  # type: ignore[arg-type]
        query_cache=query_cache,
        policy_ingestion_service=ingestion,  # type: ignore[arg-type]
        semantic_query_cache=semantic_cache,  # type: ignore[arg-type]
        rag_ops_repository=rag_ops_repository,  # type: ignore[arg-type]
        security_events_repository=security_events_repository,  # type: ignore[arg-type]
    )
    return controller, query_cache, semantic_cache, ingestion


def test_upload_policy_submits_for_quarantine_and_scan() -> None:
    """A fresh upload is quarantined and scanned, not made live - no cache
    invalidation happens here, since a quarantined document was never
    served (see test_approve_policy_bumps_corpus_version below)."""
    rag_ops_repository = _FakeRagOpsRepository()
    controller, _query_cache, semantic_cache, ingestion = _controller(
        rag_ops_repository=rag_ops_repository
    )

    response = controller.upload_policy("policy.pdf", b"content", actor="admin")

    assert response.source == "policy.pdf"
    assert response.status == "scan_passed"
    assert ingestion.submit_calls == [("policy.pdf", b"content", "admin")]
    assert rag_ops_repository.bump_calls == []
    assert semantic_cache.clear_calls == 0
    assert rag_ops_repository.invalidation_calls == 0


def test_approve_policy_bumps_corpus_version_and_clears_both_cache_layers() -> None:
    """Regression: a replaced policy must not keep serving pre-upload
    answers out of Redis/the semantic-cache pointers just because
    corpus_version isn't part of the cache key - see AdminController.approve_policy."""
    rag_ops_repository = _FakeRagOpsRepository()
    controller, query_cache, semantic_cache, ingestion = _controller(
        rag_ops_repository=rag_ops_repository
    )
    query_cache.set(CacheTier.RAG_ANSWER, "stale-question-hash", '{"answer": "old"}')

    response = controller.approve_policy("policy.pdf", actor="admin")

    assert response.source == "policy.pdf"
    assert response.status == "active"
    assert ingestion.approve_calls == [("policy.pdf", "admin")]
    assert rag_ops_repository.bump_calls == [{"actor": "admin", "source": "policy.pdf"}]
    assert semantic_cache.clear_calls == 1
    assert rag_ops_repository.invalidation_calls == 1
    assert query_cache.get(CacheTier.RAG_ANSWER, "stale-question-hash") is None


def test_approve_policy_skips_cache_invalidation_without_a_rag_ops_repository() -> None:
    controller, query_cache, semantic_cache, _ingestion = _controller(rag_ops_repository=None)
    query_cache.set(CacheTier.RAG_ANSWER, "q", '{"answer": "old"}')

    controller.approve_policy("policy.pdf", actor="admin")

    assert semantic_cache.clear_calls == 0
    assert query_cache.get(CacheTier.RAG_ANSWER, "q") == '{"answer": "old"}'


def test_reject_policy_does_not_touch_the_cache() -> None:
    rag_ops_repository = _FakeRagOpsRepository()
    controller, _query_cache, semantic_cache, ingestion = _controller(
        rag_ops_repository=rag_ops_repository
    )

    response = controller.reject_policy("policy.pdf", actor="admin", reason="prompt injection")

    assert response.status == "rejected"
    assert ingestion.reject_calls == [("policy.pdf", "admin", "prompt injection")]
    assert semantic_cache.clear_calls == 0
    assert rag_ops_repository.bump_calls == []


def test_security_events_returns_recent_sanitized_rows() -> None:
    from datetime import datetime

    event = SecurityEvent(
        id=1,
        actor="alice",
        action="input_block",
        stage="input",
        category="prompt_injection",
        mode="enforce",
        changes={"findings": []},
        reason=None,
        created_at=datetime.now(UTC),
    )
    repository = _FakeSecurityEventsRepository([event])
    controller, *_ = _controller(security_events_repository=repository)

    response = controller.security_events(limit=50)

    assert repository.list_recent_calls == [50]
    assert len(response.items) == 1
    assert response.items[0].action == "input_block"
    assert response.items[0].actor == "alice"


def test_security_events_returns_empty_without_a_repository_wired() -> None:
    controller, *_ = _controller(security_events_repository=None)

    response = controller.security_events(limit=50)

    assert response.items == []
