from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.controllers.rag_ops_controller import RagOpsController
from app.core.exceptions import InvalidRagOpsConfigError
from app.models.rag_ops import RagOpsAuditEntry, RagOpsConfig
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
        hyde_enabled=False,
        hyde_rollout_percentage=0,
        emergency_disabled=False,
        emergency_disabled_reason=None,
        emergency_disabled_at=None,
        emergency_disabled_by=None,
        corpus_version=1,
        last_cache_invalidated_at=None,
        updated_at=datetime.now(timezone.utc),
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
        updates = {k: v for k, v in kwargs.items() if v is not None and k not in {"actor", "reason"}}
        self.config = RagOpsConfig(**{**self.config.__dict__, **updates})
        return self.config

    def set_emergency_disabled(self, *, actor: str, disabled: bool, reason: str | None) -> RagOpsConfig:
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
                created_at=datetime.now(timezone.utc),
            )
        ][:limit]


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


class _FakeDynamicReranker:
    def __init__(self, has_voyage: bool = True) -> None:
        self._has_voyage = has_voyage
        self.backend: str | None = None
        self.rollout_percentage: int | None = None
        self.emergency_disabled: bool | None = None

    @property
    def has_voyage_backend(self) -> bool:
        return self._has_voyage

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


class _FakeDynamicHydeTransformer:
    def __init__(self) -> None:
        self.rollout_percentage: int | None = None
        self.emergency_disabled: bool | None = None

    def configure(self, *, rollout_percentage: int, emergency_disabled: bool) -> None:
        self.rollout_percentage = rollout_percentage
        self.emergency_disabled = emergency_disabled


def _controller(
    config: RagOpsConfig | None = None, has_voyage: bool = True
) -> tuple[
    RagOpsController,
    _FakeRepository,
    _FakeRagService,
    _FakeDynamicReranker,
    _FakeDynamicHydeTransformer,
    _FakeSemanticQueryCache,
]:
    repository = _FakeRepository(config or _config())
    rag_service = _FakeRagService()
    reranker = _FakeDynamicReranker(has_voyage=has_voyage)
    hyde_transformer = _FakeDynamicHydeTransformer()
    semantic_cache = _FakeSemanticQueryCache()
    controller = RagOpsController(
        repository, rag_service, reranker, hyde_transformer, semantic_cache, RagMetricsService()
    )
    return controller, repository, rag_service, reranker, hyde_transformer, semantic_cache


def test_status_reflects_current_config() -> None:
    controller, *_ = _controller(_config(reranking_enabled=True, corpus_version=3))

    status = controller.status()

    assert status.reranking_enabled is True
    assert status.corpus_version == 3


def test_update_config_pushes_new_values_into_live_singletons() -> None:
    controller, repository, rag_service, reranker, hyde_transformer, semantic_cache = _controller()

    controller.update_config(
        "admin",
        RagOpsConfigUpdateRequest(
            reranking_enabled=True,
            reranker_backend="voyage",
            reranker_rollout_percentage=25,
            semantic_cache_enabled=True,
            semantic_cache_threshold=0.8,
            hyde_enabled=True,
            hyde_rollout_percentage=5,
        ),
    )

    assert rag_service.reranking_enabled is True
    assert rag_service.semantic_cache_enabled is True
    assert rag_service.hyde_enabled is True
    assert reranker.backend == "voyage"
    assert reranker.rollout_percentage == 25
    assert semantic_cache.threshold == 0.8
    assert hyde_transformer.rollout_percentage == 5
    assert hyde_transformer.emergency_disabled is False


def test_update_config_rejects_voyage_backend_without_api_key() -> None:
    controller, *_ = _controller(has_voyage=False)

    with pytest.raises(InvalidRagOpsConfigError):
        controller.update_config(
            "admin", RagOpsConfigUpdateRequest(reranker_backend="voyage")
        )


def test_emergency_disable_forces_reranking_and_semantic_cache_off() -> None:
    controller, repository, rag_service, reranker, hyde_transformer, semantic_cache = _controller(
        _config(reranking_enabled=True, semantic_cache_enabled=True, hyde_enabled=True)
    )

    status = controller.emergency_disable(
        "admin", EmergencyDisableRequest(reason="incident", confirm=True)
    )

    assert status.emergency_disabled is True
    assert rag_service.reranking_enabled is False
    assert rag_service.semantic_cache_enabled is False
    assert rag_service.hyde_enabled is False
    assert reranker.emergency_disabled is True
    assert hyde_transformer.emergency_disabled is True


def test_emergency_enable_restores_previously_configured_state() -> None:
    controller, repository, rag_service, *_ = _controller(
        _config(reranking_enabled=True, emergency_disabled=True)
    )

    status = controller.emergency_enable("admin", EmergencyEnableRequest(reason=None))

    assert status.emergency_disabled is False
    assert rag_service.reranking_enabled is True


def test_audit_log_maps_repository_entries() -> None:
    controller, *_ = _controller()

    audit = controller.audit_log(10)

    assert len(audit.items) == 1
    assert audit.items[0].action == "config_update"
