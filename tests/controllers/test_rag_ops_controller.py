from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.controllers.rag_ops_controller import RagOpsController
from app.core.exceptions import InvalidRagOpsConfigError
from app.models.rag_ops import RagOpsAuditEntry, RagOpsConfig
from app.rag_services.rag_runtime_config import RagRuntimeConfigStore
from app.schemas.rag_ops import (
    EmergencyDisableRequest,
    EmergencyEnableRequest,
    RagOpsConfigUpdateRequest,
)
from app.services.rag_metrics_service import RagMetricsService


def _config(**overrides: object) -> RagOpsConfig:
    defaults: dict[str, object] = dict(
        reranking_enabled=False,
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
        updated_at=datetime.now(UTC),
        updated_by=None,
    )
    defaults.update(overrides)
    return RagOpsConfig(**defaults)  # type: ignore[arg-type]


class _FakeRepository:
    def __init__(self, config: RagOpsConfig) -> None:
        self.config = config
        self.update_calls: list[dict[str, object]] = []
        self.emergency_calls: list[dict[str, object]] = []

    def get_config(self) -> RagOpsConfig:
        return self.config

    def update_config(self, **kwargs: object) -> RagOpsConfig:
        self.update_calls.append(kwargs)
        updates = {
            k: v for k, v in kwargs.items() if v is not None and k not in {"actor", "reason"}
        }
        self.config = RagOpsConfig(**{**self.config.__dict__, **updates})
        return self.config

    def set_emergency_disabled(
        self, *, actor: str, disabled: bool, reason: str | None
    ) -> RagOpsConfig:
        self.emergency_calls.append({"actor": actor, "disabled": disabled, "reason": reason})
        self.config = RagOpsConfig(
            **{
                **self.config.__dict__,
                "emergency_disabled": disabled,
                "emergency_disabled_reason": reason,
                "emergency_disabled_by": actor if disabled else None,
            }
        )
        return self.config

    def bump_corpus_version(self, *, actor: str, source: str) -> RagOpsConfig:
        raise NotImplementedError

    def record_cache_invalidation(self) -> RagOpsConfig:
        raise NotImplementedError

    def list_audit(self, limit: int) -> list[RagOpsAuditEntry]:
        return [
            RagOpsAuditEntry(
                id=1,
                actor="admin",
                action="config_update",
                changes={"reranking_enabled": {"old": False, "new": True}},
                reason="testing",
                created_at=datetime.now(UTC),
            )
        ][:limit]


class _FakeDynamicReranker:
    def __init__(self, has_voyage: bool = True) -> None:
        self._has_voyage = has_voyage

    @property
    def has_voyage_backend(self) -> bool:
        return self._has_voyage


def _controller(
    config: RagOpsConfig | None = None, has_voyage: bool = True
) -> tuple[RagOpsController, _FakeRepository, RagRuntimeConfigStore, _FakeDynamicReranker]:
    repository = _FakeRepository(config or _config())
    config_store = RagRuntimeConfigStore()
    reranker = _FakeDynamicReranker(has_voyage=has_voyage)
    controller = RagOpsController(repository, config_store, reranker, RagMetricsService())
    return controller, repository, config_store, reranker


def test_status_reflects_current_config() -> None:
    controller, *_ = _controller(_config(reranking_enabled=True, corpus_version=3))

    status = controller.status()

    assert status.reranking_enabled is True
    assert status.corpus_version == 3


def test_update_config_pushes_new_values_into_the_shared_config_store() -> None:
    controller, repository, config_store, reranker = _controller()

    controller.update_config(
        "admin",
        RagOpsConfigUpdateRequest(
            reranking_enabled=True,
            reranker_backend="voyage",
            reranker_rollout_percentage=25,
            semantic_cache_enabled=True,
            semantic_cache_threshold=0.8,
        ),
    )

    current = config_store.current
    assert current.reranking_enabled is True
    assert current.semantic_cache_enabled is True
    assert current.reranker_backend == "voyage"
    assert current.reranker_rollout_percentage == 25
    assert current.semantic_cache_threshold == 0.8


def test_update_config_rejects_voyage_backend_without_api_key() -> None:
    controller, *_ = _controller(has_voyage=False)

    with pytest.raises(InvalidRagOpsConfigError):
        controller.update_config("admin", RagOpsConfigUpdateRequest(reranker_backend="voyage"))


def test_emergency_disable_forces_reranking_and_semantic_cache_off() -> None:
    controller, repository, config_store, reranker = _controller(
        _config(reranking_enabled=True, semantic_cache_enabled=True)
    )

    status = controller.emergency_disable(
        "admin", EmergencyDisableRequest(reason="incident", confirm=True)
    )

    assert status.emergency_disabled is True
    current = config_store.current
    assert current.reranking_enabled is False
    assert current.semantic_cache_enabled is False
    assert current.reranker_emergency_disabled is True


def test_emergency_enable_restores_previously_configured_state() -> None:
    controller, repository, config_store, _ = _controller(
        _config(reranking_enabled=True, emergency_disabled=True)
    )

    status = controller.emergency_enable("admin", EmergencyEnableRequest(reason=None))

    assert status.emergency_disabled is False
    assert config_store.current.reranking_enabled is True


def test_audit_log_maps_repository_entries() -> None:
    controller, *_ = _controller()

    audit = controller.audit_log(10)

    assert len(audit.items) == 1
    assert audit.items[0].action == "config_update"


def test_config_replacement_never_leaves_the_store_in_a_torn_state() -> None:
    """Regression: _apply() used to make six independent set_*() calls
    across three different objects - a concurrent reader between any two
    of them could observe a mix of old and new fields. Replacing the whole
    RagRuntimeConfig snapshot in one call makes that impossible: after
    update_config() returns, every field in config_store.current reflects
    the *same* update, never a partial one."""
    controller, _, config_store, _ = _controller(
        _config(reranking_enabled=False, semantic_cache_enabled=False, corpus_version=1)
    )

    controller.update_config(
        "admin",
        RagOpsConfigUpdateRequest(
            reranking_enabled=True,
            reranker_backend="voyage",
            reranker_rollout_percentage=42,
            semantic_cache_enabled=True,
            semantic_cache_threshold=0.42,
        ),
    )

    current = config_store.current
    assert (
        current.reranking_enabled,
        current.reranker_backend,
        current.reranker_rollout_percentage,
        current.semantic_cache_enabled,
        current.semantic_cache_threshold,
    ) == (True, "voyage", 42, True, 0.42)
