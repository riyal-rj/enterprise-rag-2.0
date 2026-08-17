from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from typing import Protocol

from app.rag_services.query_transformer import (
    FailOpenQueryTransformer,
    NoOpQueryTransformer,
    QueryTransformer,
    QueryTransformOutcome,
)
from app.rag_services.rollout import sampled_in


class _HyDEMetricsRecorder(Protocol):
    def record_hyde_attempt(self,*,
                            duration_ms: float,
                            fallback: bool,
                            usage_tokens: int) -> None : ...
    def record_hyde_bypass(self,*,
                           reason: str) -> None: ...


@dataclass(frozen=True)
class HyDERuntimeState:
    rollout_percentage: int
    emergency_disabled: bool

class DynamicQueryTransformer:
    def __init__(self,*,
                 delegate:QueryTransformer,
                 metrics:_HyDEMetricsRecorder,
                 rollout_percentage: int = 0,
                 emergency_disabled: bool = False) -> None:

        if not 0 <= rollout_percentage <= 100:
            raise ValueError("rollout_percentage must be between 0 and 100")

        self._delegate = FailOpenQueryTransformer(delegate)
        self._metrics=metrics
        self._lock=threading.Lock()
        self._state=HyDERuntimeState(rollout_percentage,emergency_disabled)

    def configure(self,*,
                  rollout_percentage: int,
                  emergency_disabled:bool)->None:
        if not 0 <= rollout_percentage <= 100:
            raise ValueError("rollout_percentage must be between 0 and 100")

        new_state=HyDERuntimeState(rollout_percentage,emergency_disabled)
        with self._lock:
            self._state=new_state

    def _snapshot(self)->HyDERuntimeState:
        with self._lock:
            return self._state

    @property
    def name(self)->str:
        return self._delegate.name

    @property
    def cache_namespace(self) -> str:
        state = self._snapshot()
        return (
            f"dynamic-query-transform:v1:rollout={state.rollout_percentage}"
            f":emergency={int(state.emergency_disabled)}:{self._delegate.cache_namespace}"
        )

    def transform(self,
                  query:str) -> QueryTransformOutcome:
        state = self._snapshot()
        if state.emergency_disabled:
            self._metrics.record_hyde_bypass(reason="emergency_disabled")
            return NoOpQueryTransformer(reason="emergency_disabled").transform(query)

        if not sampled_in(query,
                          state.rollout_percentage,
                          salt="hyde:v1"):
            self._metrics.record_hyde_bypass(reason="rollout")
            return NoOpQueryTransformer(reason="rollout").transform(query)

        started=time.perf_counter()
        outcome=self._delegate.transform(query)
        duration_ms=(time.perf_counter() - started) * 1000
        measured=replace(outcome, duration_ms=duration_ms)
        self._metrics.record_hyde_attempt(
            duration_ms=duration_ms,
            fallback=measured.fallback,
            usage_tokens=measured.usage_tokens
        )
        return measured
