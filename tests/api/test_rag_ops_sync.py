from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

import app.api.rag_ops_sync as rag_ops_sync
from app.models.rag_ops import RagOpsConfig


def _config(*, updated_at: datetime, reranking_enabled: bool = True) -> RagOpsConfig:
    return RagOpsConfig(
        reranking_enabled=reranking_enabled,
        reranker_backend="local",
        reranker_rollout_percentage=100,
        semantic_cache_enabled=False,
        semantic_cache_threshold=0.95,
        hyde_enabled=False,
        hyde_rollout_percentage=0,
        emergency_disabled=False,
        emergency_disabled_reason=None,
        emergency_disabled_at=None,
        emergency_disabled_by=None,
        corpus_version=1,
        last_cache_invalidated_at=None,
        updated_at=updated_at,
        updated_by=None,
    )


class _FakeRagService:
    def __init__(self) -> None:
        self.reranking_enabled: bool | None = None
        self.semantic_cache_enabled: bool | None = None
        self.hyde_enabled: bool | None = None

    def set_reranking_enabled(self, enabled: bool) -> None:
        self.reranking_enabled = enabled

    def set_semantic_cache_enabled(self, enabled: bool) -> None:
        self.semantic_cache_enabled = enabled

    def set_hyde_enabled(self, enabled: bool) -> None:
        self.hyde_enabled = enabled


class _FakeReranker:
    def __init__(self) -> None:
        self.backend: str | None = None
        self.rollout_percentage: int | None = None
        self.emergency_disabled: bool | None = None

    def set_backend(self, backend: str) -> None:
        self.backend = backend

    def set_rollout_percentage(self, percentage: int) -> None:
        self.rollout_percentage = percentage

    def set_emergency_disabled(self, disabled: bool) -> None:
        self.emergency_disabled = disabled


class _FakeSemanticQueryCache:
    def __init__(self) -> None:
        self.threshold: float | None = None

    def set_similarity_threshold(self, threshold: float) -> None:
        self.threshold = threshold


class _FakeHydeTransformer:
    def __init__(self) -> None:
        self.rollout_percentage: int | None = None
        self.emergency_disabled: bool | None = None

    def configure(self, *, rollout_percentage: int, emergency_disabled: bool) -> None:
        self.rollout_percentage = rollout_percentage
        self.emergency_disabled = emergency_disabled


class _FakeRepository:
    def __init__(self, config: RagOpsConfig) -> None:
        self.config = config
        self.get_config_calls = 0

    def get_config(self) -> RagOpsConfig:
        self.get_config_calls += 1
        return self.config


class _NotYetConstructedGetter:
    """Stand-in for an ``lru_cache``'d deps.py accessor that has never been
    called in this (fake) worker - ``cache_info().currsize == 0``. Calling
    it would defeat the laziness the poller must preserve, so it raises."""

    def cache_info(self) -> SimpleNamespace:
        return SimpleNamespace(currsize=0)

    def __call__(self) -> None:
        raise AssertionError("must not construct the singleton just to poll")


class _AlreadyConstructedGetter:
    """Stand-in for an ``lru_cache``'d deps.py accessor that some earlier
    request already caused to construct - ``cache_info().currsize == 1``,
    and calling it cheaply returns the same already-built instance."""

    def __init__(self, value: object) -> None:
        self._value = value

    def cache_info(self) -> SimpleNamespace:
        return SimpleNamespace(currsize=1)

    def __call__(self) -> object:
        return self._value


@pytest.mark.asyncio
async def test_poll_skips_entirely_when_worker_has_not_constructed_rag_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _FakeRepository(_config(updated_at=datetime.now(UTC)))
    monkeypatch.setattr(rag_ops_sync, "get_rag_service", _NotYetConstructedGetter())
    monkeypatch.setattr(rag_ops_sync, "get_rag_ops_repository", lambda: repository)

    poller = rag_ops_sync.RagOpsConfigPoller()
    await poller._poll_once()

    assert repository.get_config_calls == 0


@pytest.mark.asyncio
async def test_poll_applies_config_once_worker_has_constructed_rag_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rag_service = _FakeRagService()
    reranker = _FakeReranker()
    hyde_transformer = _FakeHydeTransformer()
    semantic_cache = _FakeSemanticQueryCache()
    updated_at = datetime.now(UTC)
    repository = _FakeRepository(_config(updated_at=updated_at, reranking_enabled=True))

    monkeypatch.setattr(rag_ops_sync, "get_rag_service", _AlreadyConstructedGetter(rag_service))
    monkeypatch.setattr(rag_ops_sync, "get_dynamic_reranker", _AlreadyConstructedGetter(reranker))
    monkeypatch.setattr(
        rag_ops_sync, "get_dynamic_hyde_transformer", _AlreadyConstructedGetter(hyde_transformer)
    )
    monkeypatch.setattr(
        rag_ops_sync, "get_semantic_query_cache", _AlreadyConstructedGetter(semantic_cache)
    )
    monkeypatch.setattr(rag_ops_sync, "get_rag_ops_repository", lambda: repository)

    poller = rag_ops_sync.RagOpsConfigPoller()
    await poller._poll_once()

    assert rag_service.reranking_enabled is True


@pytest.mark.asyncio
async def test_poll_is_a_no_op_when_config_has_not_changed_since_last_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rag_service = _FakeRagService()
    reranker = _FakeReranker()
    hyde_transformer = _FakeHydeTransformer()
    semantic_cache = _FakeSemanticQueryCache()
    updated_at = datetime.now(UTC)
    repository = _FakeRepository(_config(updated_at=updated_at, reranking_enabled=True))

    monkeypatch.setattr(rag_ops_sync, "get_rag_service", _AlreadyConstructedGetter(rag_service))
    monkeypatch.setattr(rag_ops_sync, "get_dynamic_reranker", _AlreadyConstructedGetter(reranker))
    monkeypatch.setattr(
        rag_ops_sync, "get_dynamic_hyde_transformer", _AlreadyConstructedGetter(hyde_transformer)
    )
    monkeypatch.setattr(
        rag_ops_sync, "get_semantic_query_cache", _AlreadyConstructedGetter(semantic_cache)
    )
    monkeypatch.setattr(rag_ops_sync, "get_rag_ops_repository", lambda: repository)

    poller = rag_ops_sync.RagOpsConfigPoller()
    await poller._poll_once()
    rag_service.reranking_enabled = None  # prove the second poll doesn't re-apply
    await poller._poll_once()

    assert rag_service.reranking_enabled is None


@pytest.mark.asyncio
async def test_poll_reapplies_when_config_changes_on_a_later_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rag_service = _FakeRagService()
    reranker = _FakeReranker()
    hyde_transformer = _FakeHydeTransformer()
    semantic_cache = _FakeSemanticQueryCache()
    first_updated_at = datetime.now(UTC)
    repository = _FakeRepository(_config(updated_at=first_updated_at, reranking_enabled=False))

    monkeypatch.setattr(rag_ops_sync, "get_rag_service", _AlreadyConstructedGetter(rag_service))
    monkeypatch.setattr(rag_ops_sync, "get_dynamic_reranker", _AlreadyConstructedGetter(reranker))
    monkeypatch.setattr(
        rag_ops_sync, "get_dynamic_hyde_transformer", _AlreadyConstructedGetter(hyde_transformer)
    )
    monkeypatch.setattr(
        rag_ops_sync, "get_semantic_query_cache", _AlreadyConstructedGetter(semantic_cache)
    )
    monkeypatch.setattr(rag_ops_sync, "get_rag_ops_repository", lambda: repository)

    poller = rag_ops_sync.RagOpsConfigPoller()
    await poller._poll_once()
    assert rag_service.reranking_enabled is False

    # Simulate an admin flipping the flag on another worker: the DB row's
    # updated_at moves forward.
    repository.config = _config(
        updated_at=first_updated_at + timedelta(seconds=1), reranking_enabled=True
    )
    await poller._poll_once()

    assert rag_service.reranking_enabled is True


@pytest.mark.asyncio
async def test_emergency_disable_forces_reranking_and_semantic_cache_off_via_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rag_service = _FakeRagService()
    reranker = _FakeReranker()
    hyde_transformer = _FakeHydeTransformer()
    semantic_cache = _FakeSemanticQueryCache()
    config = RagOpsConfig(
        reranking_enabled=True,
        reranker_backend="local",
        reranker_rollout_percentage=100,
        semantic_cache_enabled=True,
        semantic_cache_threshold=0.95,
        hyde_enabled=True,
        hyde_rollout_percentage=10,
        emergency_disabled=True,
        emergency_disabled_reason="incident",
        emergency_disabled_at=datetime.now(UTC),
        emergency_disabled_by="admin",
        corpus_version=1,
        last_cache_invalidated_at=None,
        updated_at=datetime.now(UTC),
        updated_by="admin",
    )
    repository = _FakeRepository(config)

    monkeypatch.setattr(rag_ops_sync, "get_rag_service", _AlreadyConstructedGetter(rag_service))
    monkeypatch.setattr(rag_ops_sync, "get_dynamic_reranker", _AlreadyConstructedGetter(reranker))
    monkeypatch.setattr(
        rag_ops_sync, "get_dynamic_hyde_transformer", _AlreadyConstructedGetter(hyde_transformer)
    )
    monkeypatch.setattr(
        rag_ops_sync, "get_semantic_query_cache", _AlreadyConstructedGetter(semantic_cache)
    )
    monkeypatch.setattr(rag_ops_sync, "get_rag_ops_repository", lambda: repository)

    poller = rag_ops_sync.RagOpsConfigPoller()
    await poller._poll_once()

    assert rag_service.reranking_enabled is False
    assert rag_service.semantic_cache_enabled is False
    assert rag_service.hyde_enabled is False
    assert hyde_transformer.emergency_disabled is True
    assert reranker.emergency_disabled is True
