from __future__ import annotations

import math
from collections.abc import Sequence

_MIN_NORM = 1e-12

def mean_pool_and_normalize(vectors:Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        raise ValueError("at least one embedding is required.")

    dimension = len(vectors[0])
    if dimension == 0:
        raise ValueError("embedding dimension must be greater than zero.")

    totals=[0.0] * dimension
    for vector_index, vector in enumerate(vectors):
        if len(vector) != dimension:
            raise ValueError(
                f"embedding {vector_index} has dimension {len(vector)}; expected {dimension}"
            )

        for value_index, raw_value in enumerate(vector):
             value=float(raw_value)
             if not math.isfinite(value):
                 raise ValueError(
                     f"embedding {vector_index} has non-finite value {raw_value} at index {value_index}"
                 )
             totals[value_index] += value

    count=float(len(vectors))
    mean=[value/count for value in totals]
    norm=math.sqrt(sum(value * value for value in mean))
    if not math.isfinite(norm) or norm <= _MIN_NORM:
        raise ValueError("mean embedding has zero or invalid norm")
    return [ value / norm for value in mean]
