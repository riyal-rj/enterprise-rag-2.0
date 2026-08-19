from __future__ import annotations

from datetime import timedelta

import pytest

import app.api.sql_proposal_recovery as sql_proposal_recovery


class _FakeRepository:
    def __init__(self, reclaimed: int = 0) -> None:
        self.reclaimed = reclaimed
        self.calls: list[timedelta] = []

    def reclaim_stale_executing(self, older_than: timedelta) -> int:
        self.calls.append(older_than)
        return self.reclaimed


async def test_reclaim_once_calls_the_repository_with_the_configured_staleness_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _FakeRepository(reclaimed=0)
    monkeypatch.setattr(sql_proposal_recovery, "get_sql_proposal_repository", lambda: repository)

    reclaimer = sql_proposal_recovery.StaleSQLProposalReclaimer(stale_after=timedelta(minutes=5))
    await reclaimer._reclaim_once()

    assert repository.calls == [timedelta(minutes=5)]


async def test_reclaim_once_does_not_raise_when_rows_are_reclaimed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _FakeRepository(reclaimed=2)
    monkeypatch.setattr(sql_proposal_recovery, "get_sql_proposal_repository", lambda: repository)

    reclaimer = sql_proposal_recovery.StaleSQLProposalReclaimer()
    await reclaimer._reclaim_once()  # must not raise

    assert repository.calls == [sql_proposal_recovery._DEFAULT_STALE_AFTER]


async def test_run_loop_survives_a_failing_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirrors RagOpsConfigPoller's own resilience guarantee: one failed
    poll must not kill the background task."""

    class _RaisingRepository:
        def reclaim_stale_executing(self, older_than: timedelta) -> int:
            del older_than
            raise RuntimeError("db unavailable")

    monkeypatch.setattr(
        sql_proposal_recovery, "get_sql_proposal_repository", lambda: _RaisingRepository()
    )

    reclaimer = sql_proposal_recovery.StaleSQLProposalReclaimer(interval_seconds=0.01)
    reclaimer.start()
    try:
        import asyncio

        await asyncio.sleep(0.05)
        assert reclaimer._task is not None
        assert not reclaimer._task.done()
    finally:
        await reclaimer.stop()


async def test_stop_before_start_is_a_no_op() -> None:
    reclaimer = sql_proposal_recovery.StaleSQLProposalReclaimer()
    await reclaimer.stop()  # must not raise
