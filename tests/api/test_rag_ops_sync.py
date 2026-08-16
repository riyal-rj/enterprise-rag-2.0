from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

import app.api.rag_ops_sync as rag_ops_sync
from app.models.rag_ops import RagOpsConfig
from app.rag_services.rag_runtime_config import RagRuntimeConfig, RagRuntimeConfigStore


def _config(*, updated_at: datetime, reranking_enabled: bool = True) -> RagOpsConfig:
    return RagOpsConfig(
        reranking_enabled=reranking_enabled,
        reranker_backend="local",
        reranker_rollout_percentage=100,
        semantic_cache_enabled=False,
        semantic_cache_threshold=0.95,
        emergency_disabled=False,
        emergency_disabled_reason=None,
        emergency_disabled_at=None,
        emergency_disabled_by=None,
        corpus_version=1,
        last_cache_invalidated_at=None,
        updated_at=updated_at,
        updated_by=None,
    )


class _FakeRepository:
    def __init__(self, config: RagOpsConfig) -> None:
        self.config = config
        self.get_config_calls = 0

    def get_config(self) -> RagOpsConfig:
        self.get_config_calls += 1
        return self.config


async def test_poll_applies_config_into_the_shared_store(monkeypatch: pytest.MonkeyPatch) -> None:
    config_store = RagRuntimeConfigStore()
    updated_at = datetime.now(UTC)
    repository = _FakeRepository(_config(updated_at=updated_at, reranking_enabled=True))

    monkeypatch.setattr(rag_ops_sync, "get_rag_runtime_config_store", lambda: config_store)
    monkeypatch.setattr(rag_ops_sync, "get_rag_ops_repository", lambda: repository)

    poller = rag_ops_sync.RagOpsConfigPoller()
    await poller._poll_once()

    assert config_store.current.reranking_enabled is True
    assert repository.get_config_calls == 1


async def test_poll_is_a_no_op_when_config_has_not_changed_since_last_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_store = RagRuntimeConfigStore()
    updated_at = datetime.now(UTC)
    repository = _FakeRepository(_config(updated_at=updated_at, reranking_enabled=True))

    monkeypatch.setattr(rag_ops_sync, "get_rag_runtime_config_store", lambda: config_store)
    monkeypatch.setattr(rag_ops_sync, "get_rag_ops_repository", lambda: repository)

    poller = rag_ops_sync.RagOpsConfigPoller()
    await poller._poll_once()
    # Replace the store's snapshot directly, out-of-band, to prove the
    # second poll doesn't touch it again (repository config unchanged).
    config_store.replace(
        RagRuntimeConfig(
            reranking_enabled=False,
            reranker_backend="local",
            reranker_rollout_percentage=100,
            reranker_emergency_disabled=False,
            semantic_cache_enabled=False,
            semantic_cache_threshold=0.95,
            corpus_version=1,
        )
    )
    await poller._poll_once()

    assert config_store.current.reranking_enabled is False
    assert repository.get_config_calls == 2


async def test_poll_reapplies_when_config_changes_on_a_later_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_store = RagRuntimeConfigStore()
    first_updated_at = datetime.now(UTC)
    repository = _FakeRepository(_config(updated_at=first_updated_at, reranking_enabled=False))

    monkeypatch.setattr(rag_ops_sync, "get_rag_runtime_config_store", lambda: config_store)
    monkeypatch.setattr(rag_ops_sync, "get_rag_ops_repository", lambda: repository)

    poller = rag_ops_sync.RagOpsConfigPoller()
    await poller._poll_once()
    assert config_store.current.reranking_enabled is False

    # Simulate an admin flipping the flag on another worker: the DB row's
    # updated_at moves forward.
    repository.config = _config(
        updated_at=first_updated_at + timedelta(seconds=1), reranking_enabled=True
    )
    await poller._poll_once()

    assert config_store.current.reranking_enabled is True


async def test_emergency_disable_forces_reranking_and_semantic_cache_off_via_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_store = RagRuntimeConfigStore()
    config = RagOpsConfig(
        reranking_enabled=True,
        reranker_backend="local",
        reranker_rollout_percentage=100,
        semantic_cache_enabled=True,
        semantic_cache_threshold=0.95,
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

    monkeypatch.setattr(rag_ops_sync, "get_rag_runtime_config_store", lambda: config_store)
    monkeypatch.setattr(rag_ops_sync, "get_rag_ops_repository", lambda: repository)

    poller = rag_ops_sync.RagOpsConfigPoller()
    await poller._poll_once()

    current = config_store.current
    assert current.reranking_enabled is False
    assert current.semantic_cache_enabled is False
    assert current.reranker_emergency_disabled is True
