"""CRAG-specific validation tests for RagOpsController.update_config.

Complements test_rag_ops_controller.py (general update/emergency-disable
coverage) - these focus on the crag_enabled/crag_web_enabled invariant
(``crag_web_enabled`` requires ``crag_enabled``) and the availability gate.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.controllers.rag_ops_controller import RagOpsController
from app.core.exceptions import InvalidRagOpsConfigError
from app.models.rag_ops import RagOpsAuditEntry, RagOpsConfig
from app.rag_services.rag_runtime_config import RagRuntimeConfigStore
from app.schemas.rag_ops import RagOpsConfigUpdateRequest
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
        crag_enabled=False,
        crag_rollout_percentage=0,
        crag_web_enabled=False,
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
        self.get_config_calls = 0

    def get_config(self) -> RagOpsConfig:
        self.get_config_calls += 1
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
        raise NotImplementedError

    def bump_corpus_version(self, *, actor: str, source: str) -> RagOpsConfig:
        raise NotImplementedError

    def record_cache_invalidation(self) -> RagOpsConfig:
        raise NotImplementedError

    def list_audit(self, limit: int) -> list[RagOpsAuditEntry]:
        return []


class _FakeDynamicReranker:
    @property
    def has_voyage_backend(self) -> bool:
        return True


def _controller(
    config: RagOpsConfig | None = None, *, crag_web_available: bool = False
) -> tuple[RagOpsController, _FakeRepository]:
    repository = _FakeRepository(config or _config())
    controller = RagOpsController(
        repository,
        RagRuntimeConfigStore(),
        _FakeDynamicReranker(),  # type: ignore[arg-type]
        RagMetricsService(),
        crag_web_available=crag_web_available,
    )
    return controller, repository


def test_enable_crag_only_succeeds() -> None:
    controller, repository = _controller()

    status = controller.update_config(
        "admin", RagOpsConfigUpdateRequest(crag_enabled=True, crag_rollout_percentage=10)
    )

    assert status.crag_enabled is True
    assert repository.config.crag_enabled is True


def test_enable_crag_and_web_together_succeeds_when_available() -> None:
    controller, repository = _controller(crag_web_available=True)

    status = controller.update_config(
        "admin", RagOpsConfigUpdateRequest(crag_enabled=True, crag_web_enabled=True)
    )

    assert status.crag_enabled is True
    assert status.crag_web_enabled is True


def test_reject_web_without_crag_in_the_same_request() -> None:
    controller, repository = _controller(crag_web_available=True)

    with pytest.raises(InvalidRagOpsConfigError):
        controller.update_config(
            "admin", RagOpsConfigUpdateRequest(crag_enabled=False, crag_web_enabled=True)
        )

    assert repository.update_calls == []


def test_reject_disabling_crag_while_web_remains_enabled() -> None:
    """A payload that only sets crag_enabled=False, leaving crag_web_enabled
    untouched (None), must still be rejected if the *current* row already
    has crag_web_enabled=True - the effective merged state is what matters,
    not just the fields this payload happens to mention."""
    controller, repository = _controller(
        _config(crag_enabled=True, crag_web_enabled=True), crag_web_available=True
    )

    with pytest.raises(InvalidRagOpsConfigError):
        controller.update_config("admin", RagOpsConfigUpdateRequest(crag_enabled=False))

    assert repository.update_calls == []


def test_disable_web_and_crag_atomically_succeeds() -> None:
    controller, repository = _controller(
        _config(crag_enabled=True, crag_web_enabled=True), crag_web_available=True
    )

    status = controller.update_config(
        "admin", RagOpsConfigUpdateRequest(crag_enabled=False, crag_web_enabled=False)
    )

    assert status.crag_enabled is False
    assert status.crag_web_enabled is False


def test_reject_web_when_capability_unavailable() -> None:
    controller, repository = _controller(crag_web_available=False)

    with pytest.raises(InvalidRagOpsConfigError):
        controller.update_config(
            "admin", RagOpsConfigUpdateRequest(crag_enabled=True, crag_web_enabled=True)
        )

    assert repository.update_calls == []


def test_enable_shadow_mode_alongside_crag_succeeds() -> None:
    controller, repository = _controller()

    status = controller.update_config(
        "admin", RagOpsConfigUpdateRequest(crag_enabled=True, crag_shadow_enabled=True)
    )

    assert status.crag_shadow_enabled is True


def test_reject_shadow_mode_without_crag() -> None:
    controller, repository = _controller()

    with pytest.raises(InvalidRagOpsConfigError):
        controller.update_config(
            "admin", RagOpsConfigUpdateRequest(crag_enabled=False, crag_shadow_enabled=True)
        )

    assert repository.update_calls == []


def test_reject_disabling_crag_while_shadow_mode_remains_enabled() -> None:
    controller, repository = _controller(_config(crag_enabled=True, crag_shadow_enabled=True))

    with pytest.raises(InvalidRagOpsConfigError):
        controller.update_config("admin", RagOpsConfigUpdateRequest(crag_enabled=False))

    assert repository.update_calls == []


def test_failed_transition_produces_no_audit_row() -> None:
    controller, repository = _controller(crag_web_available=True)

    with pytest.raises(InvalidRagOpsConfigError):
        controller.update_config(
            "admin", RagOpsConfigUpdateRequest(crag_enabled=False, crag_web_enabled=True)
        )

    assert repository.update_calls == []  # update_config() on the repository was never reached
