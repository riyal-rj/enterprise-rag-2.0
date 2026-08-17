"""Cross-worker propagation for the centralized RAG ops config.

``RagOpsController._apply`` (see ``app.controllers.rag_ops_controller``)
pushes a config change into the shared, atomically-swapped
``RagRuntimeConfigStore`` of *the worker that received the admin request*
only. Under multiple Uvicorn/Gunicorn workers, every other worker's store
keeps serving its previous snapshot until it happens to handle another
admin request itself, or restarts - the emergency kill switch in
particular would then only actually disable the feature on one worker,
while the status endpoint (which always reads straight from Postgres)
would misleadingly report it as off everywhere.

``RagOpsConfigPoller`` closes that gap: each worker runs its own background
task that periodically re-reads the config row and, if it has changed since
the last poll (``updated_at``), pushes it into that worker's own
``RagRuntimeConfigStore`` via the exact same ``apply_rag_ops_config``
function the admin-request path uses, so every worker converges on the same
config within one poll interval.

Unlike the version of this poller that pushed into ``RAGService``/
``DynamicReranker``/``QdrantSemanticQueryCache`` directly, this doesn't need
to gate on whether this worker has served a request yet: the shared
``RagRuntimeConfigStore`` (see ``app.rag_services.rag_runtime_config``) has
no Qdrant/embedding/model dependency of its own, so keeping it fresh costs
nothing regardless of whether the heavier RAG-serving singletons that read
it have been lazily constructed yet.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from app.api.deps import get_rag_ops_repository, get_rag_runtime_config_store
from app.controllers.rag_ops_controller import apply_rag_ops_config

logger = logging.getLogger(__name__)

_DEFAULT_POLL_INTERVAL_SECONDS = 5.0


class RagOpsConfigPoller:
    """Background asyncio task: re-polls ``RagOpsRepository.get_config()``
    on an interval and re-applies it to this worker's ``RagRuntimeConfigStore``
    whenever ``updated_at`` has moved past the last-seen value - see module
    docstring.
    """

    def __init__(self, *, interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS) -> None:
        self._interval_seconds = interval_seconds
        self._last_seen_updated_at: datetime | None = None
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="rag-ops-config-poller")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._interval_seconds)
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a poll failure must not kill the background task
                logger.exception("rag_ops.config_poll_failed")

    async def _poll_once(self) -> None:
        config = await asyncio.to_thread(get_rag_ops_repository().get_config)
        if (
            self._last_seen_updated_at is not None
            and config.updated_at <= self._last_seen_updated_at
        ):
            return
        self._last_seen_updated_at = config.updated_at

        apply_rag_ops_config(config, config_store=get_rag_runtime_config_store())
        logger.info(
            "rag_ops.config_synced_from_poll",
            extra={"updated_at": config.updated_at.isoformat()},
        )
