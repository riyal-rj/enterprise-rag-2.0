"""Embedding-fusion package: combines multiple hypothesis embeddings (e.g.
HyDE's) into one retrieval vector."""

from __future__ import annotations

from app.rag_services.fusion.embedding_fusion import (
    FUSION_ALGORITHM_VERSION,
    mean_pool_and_normalize,
)

__all__ = [
    "FUSION_ALGORITHM_VERSION",
    "mean_pool_and_normalize",
]
