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
    see module docstring.

    ``emergency_disabled`` is the single global kill switch - it isn't
    reranker-specific despite the historical, reranker-prefixed field name
    it replaces; reranking/semantic-cache gate on it pre-baked into their
    own ``*_enabled`` fields below, while HyDE's
    ``hyde_enabled`` is intentionally left raw (not pre-baked) so
    ``DynamicQueryTransformer.plan`` can distinguish "feature off" from
    "emergency-disabled" and keep both bypass reasons observable in
    ``HyDEMetricsSnapshot`` (see that module and ``RAGService.answer``).
    """

    reranking_enabled: bool
    reranker_backend: RerankerBackendName
    reranker_rollout_percentage: int
    emergency_disabled: bool
    semantic_cache_enabled: bool
    semantic_cache_threshold: float
    corpus_version: int
    hyde_enabled: bool
    hyde_rollout_percentage: int
    crag_enabled: bool
    crag_rollout_percentage: int
    crag_web_enabled: bool
    # Observe-only rollout stage, ahead of a real (web-eligible) treatment
    # rollout - see DynamicCorrectiveRetriever.plan and RAGService.answer's
    # "shadow" cohort handling. Defaulted so every pre-existing construction
    # site (deps.py, RagOpsController, tests) keeps working unchanged.
    crag_shadow_enabled: bool = False
    # Self-reflective answer critique/revision stage - see
    # app.rag_services.reflection.dynamic_self_reflection and
    # RAGService.answer's post-answer reflection block. Two-field shape
    # (enabled/rollout%), same as HyDE - no web/shadow-style dependent
    # sub-flag exists for this stage. Defaulted so every pre-existing
    # construction site keeps working unchanged.
    self_reflective_enabled: bool = False
    self_reflective_rollout_percentage: int = 0
    # Text-to-SQL routing (app.query_orchestration.query_orchestrator) -
    # admin-only, proposal-only. Defaulted so every pre-existing
    # construction site keeps working unchanged, same reasoning as
    # crag_shadow_enabled/self_reflective_enabled above.
    sql_enabled: bool = False
    sql_rollout_percentage: int = 0
    sql_proposal_only: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.reranker_rollout_percentage <= 100:
            raise ValueError("reranker_rollout_percentage must be between 0 and 100")
        if not 0 <= self.hyde_rollout_percentage <= 100:
            raise ValueError("hyde_rollout_percentage must be between 0 and 100")
        if not 0.0 <= self.semantic_cache_threshold <= 1.0:
            raise ValueError("semantic_cache_threshold must be between 0.0 and 1.0")
        if self.corpus_version < 1:
            raise ValueError("corpus_version must be positive")
        if not 0 <= self.crag_rollout_percentage <= 100:
            raise ValueError("crag_rollout_percentage must be between 0 and 100")
        if self.crag_web_enabled and not self.crag_enabled:
            raise ValueError("crag_web_enabled requires crag_enabled")
        if self.crag_shadow_enabled and not self.crag_enabled:
            raise ValueError("crag_shadow_enabled requires crag_enabled")
        if not 0 <= self.self_reflective_rollout_percentage <= 100:
            raise ValueError("self_reflective_rollout_percentage must be between 0 and 100")
        if not 0 <= self.sql_rollout_percentage <= 100:
            raise ValueError("sql_rollout_percentage must be between 0 and 100")
        if not self.sql_proposal_only:
            # Non-negotiable for this release (see the architecture
            # blueprint's "Non-negotiable controls" #7 and rollout gate #9):
            # there is no automatic-execution code path at all, so nothing
            # upstream (an admin request, a bad default, a bug) can ever
            # construct a config that would try to use one.
            raise ValueError(
                "sql_proposal_only cannot be disabled - this release has no "
                "automatic SQL execution path"
            )


_DEFAULT_CONFIG = RagRuntimeConfig(
    reranking_enabled=False,
    reranker_backend="local",
    reranker_rollout_percentage=100,
    emergency_disabled=False,
    semantic_cache_enabled=False,
    semantic_cache_threshold=0.95,
    corpus_version=1,
    hyde_enabled=False,
    hyde_rollout_percentage=0,
    crag_enabled=False,
    crag_rollout_percentage=0,
    crag_web_enabled=False,
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
