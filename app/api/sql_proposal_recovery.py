"""Background recovery for SQL proposals stuck in ``EXECUTING``.

``SQLProposalRepository.lock_for_execution`` commits ``status='executing'``
in its own transaction before ``SQLService.approve_and_execute`` does
anything else. A process crash between that commit and the terminal
``mark_executed``/``mark_failed`` call leaves the row in ``EXECUTING``
forever - only ``PROPOSED`` proposals can ever be locked for execution
again, so nothing else would ever move it out of that state. This mirrors
``app.api.rag_ops_sync.RagOpsConfigPoller`` - the one other background-task
pattern in this codebase (a plain ``asyncio.create_task`` loop with
``start()``/``stop()``, swallowing per-iteration exceptions so one failed
poll doesn't kill the task) - rather than introducing new job
infrastructure for a single periodic query.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

from app.api.deps import get_sql_proposal_repository

logger = logging.getLogger(__name__)

_DEFAULT_POLL_INTERVAL_SECONDS = 60.0
# Well beyond the executor's own statement_timeout - this only ever catches
# a genuinely abandoned EXECUTING row (a crashed worker), never a query
# that's merely running long within its own configured limit.
_DEFAULT_STALE_AFTER = timedelta(minutes=5)


class StaleSQLProposalReclaimer:
    def __init__(
        self,
        *,
        interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        stale_after: timedelta = _DEFAULT_STALE_AFTER,
    ) -> None:
        self._interval_seconds = interval_seconds
        self._stale_after = stale_after
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="stale-sql-proposal-reclaimer")

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
                await self._reclaim_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a poll failure must not kill the background task
                logger.exception("sql.stale_proposal_reclaim_failed")

    async def _reclaim_once(self) -> None:
        reclaimed = await asyncio.to_thread(
            get_sql_proposal_repository().reclaim_stale_executing, self._stale_after
        )
        if reclaimed:
            logger.warning("sql.stale_proposals_reclaimed", extra={"count": reclaimed})
