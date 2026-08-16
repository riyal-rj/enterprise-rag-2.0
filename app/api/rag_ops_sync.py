"""Cross-worker propagation for the centralized RAG ops config.

``RagOpsController._apply`` (see ``app.controllers.rag_ops_controller``)
pushes a config change into the live singletons of *the worker that
received the admin request* only. Under multiple Uvicorn/Gunicorn workers,
every other worker keeps serving its previous in-memory
reranking/semantic-cache/kill-switch state until it happens to handle
another admin request itself, or restarts - the emergency kill switch in
particular would then only actually disable the feature on one worker,
while the status endpoint (which always reads straight from Postgres)
would misleadingly report it as off everywhere.

``RagOpsConfigPoller`` closes that gap: each worker runs its own background
task that periodically re-reads the config row and, if it has changed since
the last poll (``updated_at``), pushes it into that worker's own singletons
via the exact same ``apply_rag_ops_config`` function the admin-request path
uses, so every worker converges on the same config within one poll
interval.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from app.api.deps import (
    get_dynamic_reranker,
    get_rag_ops_repository,
    get_rag_service,
    get_semantic_query_cache,
)
from app.controllers.rag_ops_controller import apply_rag_ops_config

logger = logging.getLogger(__name__)

_DEFAULT_POLL_INTERVAL_SECONDS = 5.0


class RagOpsConfigPoller:
    """Background asyncio task: re-polls ``RagOpsRepository.get_config()``
    on an interval and re-applies it to this worker's singletons whenever
    ``updated_at`` has moved past the last-seen value - see module
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
        # This worker hasn't lazily constructed the RAG-serving singletons
        # yet (no /chat request has landed here) - nothing to keep in sync,
        # and calling get_rag_service() here would force the expensive
        # embedding-client/reranker-model/Qdrant construction that
        # laziness exists specifically to defer (see
        # app.api.deps.get_dynamic_reranker, get_rag_service). Once a
        # worker *has* served a request, get_rag_service() having
        # constructed it also guarantees get_dynamic_reranker() and
        # get_semantic_query_cache() were constructed too, since
        # get_rag_service() calls both internally.
        if get_rag_service.cache_info().currsize == 0:
            return

        config = await asyncio.to_thread(get_rag_ops_repository().get_config)
        if (
            self._last_seen_updated_at is not None
            and config.updated_at <= self._last_seen_updated_at
        ):
            return
        self._last_seen_updated_at = config.updated_at

        apply_rag_ops_config(
            config,
            rag_service=get_rag_service(),
            reranker=get_dynamic_reranker(),
            semantic_query_cache=get_semantic_query_cache(),
        )
        logger.info(
            "rag_ops.config_synced_from_poll",
            extra={"updated_at": config.updated_at.isoformat()},
        )
