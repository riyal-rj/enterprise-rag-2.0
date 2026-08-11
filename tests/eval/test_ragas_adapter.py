"""``ragas_adapter.run`` needs a live LLM to actually score rows, so it
isn't exercised end-to-end here. This proves two things instead: the
module stays importable even when the ``ragas`` package's own import chain
is broken (the import is deferred to inside ``run``), and the
retry/timeout wrapper around ``ragas.evaluate`` behaves correctly given a
fake ``evaluate_fn`` — the real ``ragas.evaluate`` can't be exercised here
since ``ragas`` doesn't import cleanly in this environment.
"""

from __future__ import annotations

import time

import pytest

from app.eval import ragas_adapter


def test_module_imports_without_pulling_in_ragas() -> None:
    assert callable(ragas_adapter.run)
    assert ragas_adapter.METRIC_NAMES == (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    )


def test_evaluate_with_resilience_returns_on_first_success() -> None:
    calls: list[int] = []

    def fake_evaluate(dataset: object, metrics: object) -> str:
        calls.append(1)
        return "ok"

    result = ragas_adapter._evaluate_with_resilience(object(), [], evaluate_fn=fake_evaluate)

    assert result == "ok"
    assert len(calls) == 1


def test_evaluate_with_resilience_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ragas_adapter, "_RETRY_BACKOFF_SECONDS", 0.001)
    attempts: list[int] = []

    def flaky_evaluate(dataset: object, metrics: object) -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("transient")
        return "ok"

    result = ragas_adapter._evaluate_with_resilience(object(), [], evaluate_fn=flaky_evaluate)

    assert result == "ok"
    assert len(attempts) == 3


def test_evaluate_with_resilience_raises_after_exhausting_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ragas_adapter, "_RETRY_BACKOFF_SECONDS", 0.001)

    def always_fails(dataset: object, metrics: object) -> str:
        raise RuntimeError("permanent failure")

    with pytest.raises(RuntimeError, match="permanent failure"):
        ragas_adapter._evaluate_with_resilience(object(), [], evaluate_fn=always_fails)


def test_evaluate_with_resilience_times_out_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hung first attempt is abandoned at the timeout and retried — and
    doesn't block on the leaked thread finishing (the bug the ``wait=False``
    shutdown fixes)."""
    monkeypatch.setattr(ragas_adapter, "_RETRY_BACKOFF_SECONDS", 0.001)
    monkeypatch.setattr(ragas_adapter, "_EVALUATE_TIMEOUT_SECONDS", 0.05)
    attempts: list[int] = []

    def slow_then_fast(dataset: object, metrics: object) -> str:
        attempts.append(1)
        if len(attempts) == 1:
            time.sleep(0.3)  # exceeds the 0.05s timeout
        return "ok"

    started = time.monotonic()
    result = ragas_adapter._evaluate_with_resilience(object(), [], evaluate_fn=slow_then_fast)
    elapsed = time.monotonic() - started

    assert result == "ok"
    assert len(attempts) == 2
    assert elapsed < 0.3  # would be ~0.3s+ if shutdown blocked on the hung thread
