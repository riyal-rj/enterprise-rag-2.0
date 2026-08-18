from __future__ import annotations

from datetime import UTC

from app.controllers.admin_controller import AdminController
from app.core.config.cache import CacheSettings
from app.models.rag_ops import RagOpsAuditEntry, RagOpsConfig
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


class _FakePolicyIngestionService:
    def __init__(self) -> None:
        self.ingest_calls: list[tuple[str, bytes]] = []

    def list_policies(self) -> PolicyListResponse:
        raise NotImplementedError

    def ingest(self, filename: str, content: bytes) -> PolicyUploadResponse:
        self.ingest_calls.append((filename, content))
        return PolicyUploadResponse(source=filename, chunks_ingested=3, replaced=False)


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


def _controller(
    *, rag_ops_repository: _FakeRagOpsRepository | None = None
) -> tuple[
    AdminController, QueryCacheService, _FakeSemanticQueryCache, _FakePolicyIngestionService
]:
    query_cache = QueryCacheService(_InMemoryCacheBackend(), CacheSettings())
    semantic_cache = _FakeSemanticQueryCache()
    ingestion = _FakePolicyIngestionService()
    controller = AdminController(
        health_check_service=None,  # type: ignore[arg-type]
        query_cache=query_cache,
        policy_ingestion_service=ingestion,  # type: ignore[arg-type]
        semantic_query_cache=semantic_cache,  # type: ignore[arg-type]
        rag_ops_repository=rag_ops_repository,  # type: ignore[arg-type]
    )
    return controller, query_cache, semantic_cache, ingestion


def test_upload_policy_bumps_corpus_version_and_clears_both_cache_layers() -> None:
    """Regression: a replaced policy must not keep serving pre-upload
    answers out of Redis/the semantic-cache pointers just because
    corpus_version isn't part of the cache key - see AdminController.upload_policy."""
    rag_ops_repository = _FakeRagOpsRepository()
    controller, query_cache, semantic_cache, ingestion = _controller(
        rag_ops_repository=rag_ops_repository
    )
    query_cache.set(CacheTier.RAG_ANSWER, "stale-question-hash", '{"answer": "old"}')

    response = controller.upload_policy("policy.pdf", b"content", actor="admin")

    assert response.source == "policy.pdf"
    assert ingestion.ingest_calls == [("policy.pdf", b"content")]
    assert rag_ops_repository.bump_calls == [{"actor": "admin", "source": "policy.pdf"}]
    assert semantic_cache.clear_calls == 1
    assert rag_ops_repository.invalidation_calls == 1
    assert query_cache.get(CacheTier.RAG_ANSWER, "stale-question-hash") is None


def test_upload_policy_skips_cache_invalidation_without_a_rag_ops_repository() -> None:
    controller, query_cache, semantic_cache, ingestion = _controller(rag_ops_repository=None)
    query_cache.set(CacheTier.RAG_ANSWER, "q", '{"answer": "old"}')

    controller.upload_policy("policy.pdf", b"content", actor="admin")

    assert semantic_cache.clear_calls == 0
    assert query_cache.get(CacheTier.RAG_ANSWER, "q") == '{"answer": "old"}'
