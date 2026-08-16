"""Atomic snapshot of the admin-mutable RAG Ops runtime config.

Before this module existed, ``RagOpsController._apply``/``RagOpsConfigPoller``
pushed a config change into three *separately* mutable objects
(``RAGService``, ``DynamicReranker``, ``QdrantSemanticQueryCache``) via six
independent ``set_*()`` calls. Nothing serialized those six writes against a
concurrent request reading the fields mid-sequence, so a request racing an
admin's config update could observe a torn mix - e.g. ``DynamicReranker``
with its new ``backend`` already applied but its old ``emergency_disabled``
still in effect, or vice versa.

``RagRuntimeConfig`` bundles every one of those fields into a single frozen
snapshot; ``RagRuntimeConfigStore`` holds exactly one such snapshot and
replaces it with one atomic reference swap. A reader that does
``store.current`` once, up front, and works from that local reference for
the rest of its call (see ``RAGService.answer``, ``DynamicReranker.rerank``)
either sees the whole pre-update config or the whole post-update config,
never a mix - regardless of how many fields a future admin-mutable flag
(e.g. HyDE's) adds to the snapshot.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal

RerankerBackendName = Literal["local", "voyage"]


@dataclass(frozen=True)
class RagRuntimeConfig:
    """Every RAG Ops field that a live request path reads mid-pipeline -
    see module docstring."""

    reranking_enabled: bool
    reranker_backend: RerankerBackendName
    reranker_rollout_percentage: int
    reranker_emergency_disabled: bool
    semantic_cache_enabled: bool
    semantic_cache_threshold: float
    corpus_version: int


_DEFAULT_CONFIG = RagRuntimeConfig(
    reranking_enabled=False,
    reranker_backend="local",
    reranker_rollout_percentage=100,
    reranker_emergency_disabled=False,
    semantic_cache_enabled=False,
    semantic_cache_threshold=0.95,
    corpus_version=1,
)


class RagRuntimeConfigStore:
    """Holds one :class:`RagRuntimeConfig`, replaced as a whole under a
    lock. The lock serializes concurrent *writers* (an admin request racing
    ``RagOpsConfigPoller``); it isn't needed for atomicity of a single
    ``current`` read against a single ``replace`` write - a bare attribute
    read/write is already atomic under CPython's GIL - but it does make the
    "replace the whole snapshot, never partially" intent explicit and
    future-proofs against a non-CPython runtime.
    """

    def __init__(self, initial: RagRuntimeConfig = _DEFAULT_CONFIG) -> None:
        self._lock = threading.Lock()
        self._current = initial

    @property
    def current(self) -> RagRuntimeConfig:
        return self._current

    def replace(self, config: RagRuntimeConfig) -> None:
        with self._lock:
            self._current = config
