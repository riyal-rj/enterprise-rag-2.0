from __future__ import annotations

import math

import pytest

from app.rag_services.embedding_fusion import mean_pool_and_normalize


def test_mean_pool_and_normalize_returns_unit_vector() -> None:
    result = mean_pool_and_normalize([[1.0, 0.0], [0.0, 1.0]])

    assert result == pytest.approx([2**-0.5, 2**-0.5])
    assert math.sqrt(sum(value * value for value in result)) == pytest.approx(1.0)


def test_mean_pool_and_normalize_single_vector_is_just_normalized() -> None:
    result = mean_pool_and_normalize([[3.0, 4.0]])

    assert result == pytest.approx([0.6, 0.8])


@pytest.mark.parametrize(
    "vectors",
    [
        [],
        [[]],
        [[1.0], [1.0, 2.0]],
        [[float("nan")]],
        [[float("inf")]],
        [[0.0, 0.0]],
    ],
)
def test_mean_pool_and_normalize_rejects_invalid_vectors(vectors: list[list[float]]) -> None:
    with pytest.raises(ValueError):
        mean_pool_and_normalize(vectors)


def test_mean_pool_and_normalize_cancelling_vectors_have_zero_norm() -> None:
    with pytest.raises(ValueError):
        mean_pool_and_normalize([[1.0, 0.0], [-1.0, 0.0]])
