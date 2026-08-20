"""Hand-written scanner fakes implementing the real Protocols from
``app.guardrails.contracts``.

Deliberately not ``unittest.mock`` - matching this repo's house style (see
``_FakeReranker``/``_FakeLLMClient`` in ``tests/rag_services/test_rag_service.py``)
and, more importantly, keeping every guardrail test free of any real
``llm_guard`` import so the default test run never downloads a transformer
model. ``tests/guardrails/test_llm_guard_adapter.py`` is the only file that
touches the real library, and it's marked ``slow``.
"""

from __future__ import annotations

from app.guardrails.contracts import ScanFinding


class FakeTextScanner:
    """Returns a canned finding tuple, recording every text it was given."""

    def __init__(
        self, findings: tuple[ScanFinding, ...] = (), *, name: str = "fake.text"
    ) -> None:
        self._findings = findings
        self._name = name
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def scan(self, text: str) -> tuple[ScanFinding, ...]:
        self.calls.append(text)
        return self._findings


class FakeOutputScanner:
    """Output-side equivalent of :class:`FakeTextScanner`."""

    def __init__(
        self, findings: tuple[ScanFinding, ...] = (), *, name: str = "fake.output"
    ) -> None:
        self._findings = findings
        self._name = name
        self.calls: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return self._name

    def scan(self, *, prompt: str, output: str) -> tuple[ScanFinding, ...]:
        self.calls.append((prompt, output))
        return self._findings


class RaisingTextScanner:
    """Simulates a scanner blowing up mid-scan (model/runtime failure)."""

    def __init__(self, *, name: str = "fake.raising") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def scan(self, text: str) -> tuple[ScanFinding, ...]:
        del text
        raise RuntimeError("scanner exploded")
