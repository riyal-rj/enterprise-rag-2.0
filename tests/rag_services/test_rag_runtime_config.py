from __future__ import annotations

import pytest

from app.rag_services.rag_runtime_config import RagRuntimeConfig, RagRuntimeConfigStore


def _valid_kwargs(**overrides: object) -> dict[str, object]:
    defaults: dict[str, object] = dict(
        reranking_enabled=True,
        reranker_backend="local",
        reranker_rollout_percentage=50,
        emergency_disabled=False,
        semantic_cache_enabled=True,
        semantic_cache_threshold=0.9,
        corpus_version=3,
        hyde_enabled=True,
        hyde_rollout_percentage=25,
        crag_enabled=False,
        crag_rollout_percentage=0,
        crag_web_enabled=False,
    )
    defaults.update(overrides)
    return defaults


def test_valid_snapshot_constructs_without_error() -> None:
    config = RagRuntimeConfig(**_valid_kwargs())  # type: ignore[arg-type]

    assert config.reranker_rollout_percentage == 50
    assert config.hyde_rollout_percentage == 25


@pytest.mark.parametrize("value", [-1, 101, 1000])
def test_invalid_reranker_rollout_percentage_is_rejected(value: int) -> None:
    with pytest.raises(ValueError, match="reranker_rollout_percentage"):
        RagRuntimeConfig(**_valid_kwargs(reranker_rollout_percentage=value))  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-1, 101, 1000])
def test_invalid_hyde_rollout_percentage_is_rejected(value: int) -> None:
    with pytest.raises(ValueError, match="hyde_rollout_percentage"):
        RagRuntimeConfig(**_valid_kwargs(hyde_rollout_percentage=value))  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-0.01, 1.01, 5.0])
def test_invalid_semantic_cache_threshold_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="semantic_cache_threshold"):
        RagRuntimeConfig(**_valid_kwargs(semantic_cache_threshold=value))  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1, -100])
def test_invalid_corpus_version_is_rejected(value: int) -> None:
    with pytest.raises(ValueError, match="corpus_version"):
        RagRuntimeConfig(**_valid_kwargs(corpus_version=value))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reranker_rollout_percentage", 0),
        ("reranker_rollout_percentage", 100),
        ("hyde_rollout_percentage", 0),
        ("hyde_rollout_percentage", 100),
        ("semantic_cache_threshold", 0.0),
        ("semantic_cache_threshold", 1.0),
        ("corpus_version", 1),
    ],
)
def test_boundary_values_are_accepted(field: str, value: object) -> None:
    RagRuntimeConfig(**_valid_kwargs(**{field: value}))  # type: ignore[arg-type]


def test_store_replace_is_a_single_atomic_swap() -> None:
    store = RagRuntimeConfigStore(RagRuntimeConfig(**_valid_kwargs(corpus_version=1)))  # type: ignore[arg-type]

    store.replace(RagRuntimeConfig(**_valid_kwargs(corpus_version=2)))  # type: ignore[arg-type]

    assert store.current.corpus_version == 2


def test_store_default_construction_uses_a_safe_default_snapshot() -> None:
    store = RagRuntimeConfigStore()

    config = store.current
    assert config.reranking_enabled is False
    assert config.semantic_cache_enabled is False
    assert config.hyde_enabled is False
    assert config.emergency_disabled is False
    assert config.crag_shadow_enabled is False


def test_crag_shadow_enabled_defaults_to_false_for_existing_call_sites() -> None:
    config = RagRuntimeConfig(**_valid_kwargs())  # type: ignore[arg-type]

    assert config.crag_shadow_enabled is False


def test_crag_shadow_enabled_requires_crag_enabled() -> None:
    with pytest.raises(ValueError, match="crag_shadow_enabled"):
        RagRuntimeConfig(
            **_valid_kwargs(crag_enabled=False, crag_shadow_enabled=True)  # type: ignore[arg-type]
        )


def test_crag_shadow_enabled_is_accepted_alongside_crag_enabled() -> None:
    config = RagRuntimeConfig(
        **_valid_kwargs(crag_enabled=True, crag_shadow_enabled=True)  # type: ignore[arg-type]
    )

    assert config.crag_shadow_enabled is True


def test_sql_fields_default_to_disabled_and_proposal_only_for_existing_call_sites() -> None:
    config = RagRuntimeConfig(**_valid_kwargs())  # type: ignore[arg-type]

    assert config.sql_enabled is False
    assert config.sql_rollout_percentage == 0
    assert config.sql_proposal_only is True


@pytest.mark.parametrize("value", [-1, 101, 1000])
def test_invalid_sql_rollout_percentage_is_rejected(value: int) -> None:
    with pytest.raises(ValueError, match="sql_rollout_percentage"):
        RagRuntimeConfig(**_valid_kwargs(sql_rollout_percentage=value))  # type: ignore[arg-type]


def test_sql_proposal_only_cannot_be_disabled() -> None:
    """Non-negotiable for this release (see the Text-to-SQL architecture
    blueprint's controls #7/#9): there is no automatic-execution code path,
    so nothing can ever construct a config that would try to use one."""
    with pytest.raises(ValueError, match="sql_proposal_only"):
        RagRuntimeConfig(**_valid_kwargs(sql_proposal_only=False))  # type: ignore[arg-type]


def test_store_default_construction_disables_sql_and_keeps_proposal_only() -> None:
    store = RagRuntimeConfigStore()

    config = store.current
    assert config.sql_enabled is False
    assert config.sql_proposal_only is True
