from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import httpx
import pytest
from openai import (
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel

from app.core.llm import chat_client
from app.core.llm.chat_client import OpenAILLMClient, build_openai_client


class _Answer(BaseModel):
    value: str


class _AnswerWithOptionalField(BaseModel):
    value: str
    note: str | None = None


_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _rate_limit_error(message: str = "rate limited") -> RateLimitError:
    return RateLimitError(message, response=httpx.Response(429, request=_REQUEST), body=None)


def _server_error(message: str = "server error") -> InternalServerError:
    return InternalServerError(message, response=httpx.Response(500, request=_REQUEST), body=None)


def _bad_request_error(message: str = "bad request") -> BadRequestError:
    return BadRequestError(message, response=httpx.Response(400, request=_REQUEST), body=None)


def _auth_error(message: str = "invalid api key") -> AuthenticationError:
    return AuthenticationError(message, response=httpx.Response(401, request=_REQUEST), body=None)


@dataclass
class _FakeMessage:
    content: str | None


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class _FakeChatCompletion:
    choices: list[_FakeChoice]
    usage: _FakeUsage | None = None


def _fake_completion(text: str, usage: _FakeUsage | None = None) -> _FakeChatCompletion:
    return _FakeChatCompletion(
        choices=[_FakeChoice(message=_FakeMessage(content=text))], usage=usage
    )


class _FakeCompletions:
    """Test double for ``client.chat.completions``. Pops from a queue of
    canned responses/exceptions and records every call's kwargs."""

    def __init__(self, queue: list[_FakeChatCompletion | Exception]) -> None:
        self._queue = list(queue)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _FakeChatCompletion:
        self.calls.append(kwargs)
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeOpenAI:
    def __init__(self, queue: list[_FakeChatCompletion | Exception]) -> None:
        self.completions = _FakeCompletions(queue)
        self.chat = _FakeChat(self.completions)


def _client(*queue: _FakeChatCompletion | Exception) -> tuple[OpenAILLMClient, _FakeOpenAI]:
    fake = _FakeOpenAI(list(queue))
    return (
        OpenAILLMClient(
            client=cast(OpenAI, fake),
            default_answer_model="gpt-4o",
            default_grader_model="gpt-4o-mini",
        ),
        fake,
    )


def test_build_openai_client_rejects_empty_api_key() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        build_openai_client("")


def test_build_openai_client_returns_real_openai_instance() -> None:
    client = build_openai_client("sk-test-key")
    assert isinstance(client, OpenAI)


def test_generate_returns_text_and_usage() -> None:
    llm, fake = _client(_fake_completion("hello", _FakeUsage(10, 5, 15)))

    result = llm.generate("system", "user")

    assert result.text == "hello"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 15
    assert "response_format" not in fake.completions.calls[0]


def test_generate_uses_default_answer_model_when_none_given() -> None:
    llm, fake = _client(_fake_completion("hello"))

    llm.generate("system", "user")

    assert fake.completions.calls[0]["model"] == "gpt-4o"


def test_generate_with_explicit_model_overrides_default() -> None:
    llm, fake = _client(_fake_completion("hello"))

    llm.generate("system", "user", model="gpt-4o-mini-custom")

    assert fake.completions.calls[0]["model"] == "gpt-4o-mini-custom"


def test_generate_json_uses_grader_model_and_sets_response_format() -> None:
    llm, fake = _client(_fake_completion('{"a": 1}'))

    result = llm.generate_json("system", "user")

    assert result.text == '{"a": 1}'
    assert fake.completions.calls[0]["model"] == "gpt-4o-mini"
    assert fake.completions.calls[0]["response_format"] == {"type": "json_object"}


def test_missing_usage_defaults_to_zero() -> None:
    llm, _ = _client(_fake_completion("hello", usage=None))

    result = llm.generate("system", "user")

    assert result.usage.prompt_tokens == 0
    assert result.usage.completion_tokens == 0
    assert result.usage.total_tokens == 0


def test_missing_content_defaults_to_empty_string() -> None:
    llm, _ = _client(_fake_completion(text=None))  # type: ignore[arg-type]

    result = llm.generate("system", "user")

    assert result.text == ""


def test_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_client, "_RETRY_BACKOFF_SECONDS", 0.001)
    llm, fake = _client(_rate_limit_error(), _fake_completion("recovered"))

    result = llm.generate("system", "user")

    assert result.text == "recovered"
    assert len(fake.completions.calls) == 2


def test_raises_after_exhausting_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_client, "_RETRY_BACKOFF_SECONDS", 0.001)
    llm, fake = _client(
        _rate_limit_error("fail 1"), _rate_limit_error("fail 2"), _rate_limit_error("fail 3")
    )

    with pytest.raises(RateLimitError, match="fail 3"):
        llm.generate("system", "user")

    assert len(fake.completions.calls) == 3


def test_server_error_is_also_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_client, "_RETRY_BACKOFF_SECONDS", 0.001)
    llm, fake = _client(_server_error(), _fake_completion("recovered"))

    result = llm.generate("system", "user")

    assert result.text == "recovered"
    assert len(fake.completions.calls) == 2


def test_permanent_error_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_client, "_RETRY_BACKOFF_SECONDS", 0.001)
    llm, fake = _client(_bad_request_error("malformed request"), _fake_completion("never reached"))

    with pytest.raises(BadRequestError, match="malformed request"):
        llm.generate("system", "user")

    # Exactly one attempt - a permanent failure fails fast, it doesn't burn
    # the retry budget on an error that will look identical every time.
    assert len(fake.completions.calls) == 1


def test_authentication_error_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_client, "_RETRY_BACKOFF_SECONDS", 0.001)
    llm, fake = _client(_auth_error(), _fake_completion("never reached"))

    with pytest.raises(AuthenticationError):
        llm.generate("system", "user")

    assert len(fake.completions.calls) == 1


class _FakeClock:
    """Controllable monotonic clock for deadline tests - avoids actually
    sleeping/waiting real wall-clock time in the test suite."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class _TimeAdvancingCompletions:
    """Like _FakeCompletions, but each queued item also advances the fake
    clock by a caller-chosen amount before returning/raising - simulates a
    call that itself took real wall-clock time."""

    def __init__(
        self, clock: _FakeClock, queue: list[tuple[float, _FakeChatCompletion | Exception]]
    ) -> None:
        self._clock = clock
        self._queue = list(queue)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _FakeChatCompletion:
        self.calls.append(kwargs)
        elapsed, item = self._queue.pop(0)
        self._clock.now += elapsed
        if isinstance(item, Exception):
            raise item
        return item


class _FakeOpenAIWithClock:
    def __init__(self, completions: _TimeAdvancingCompletions) -> None:
        self.chat = _FakeChat(cast(_FakeCompletions, completions))


def _client_with_clock(
    clock: _FakeClock, queue: list[tuple[float, _FakeChatCompletion | Exception]]
) -> tuple[OpenAILLMClient, _TimeAdvancingCompletions]:
    completions = _TimeAdvancingCompletions(clock, queue)
    fake = _FakeOpenAIWithClock(completions)
    return (
        OpenAILLMClient(
            client=cast(OpenAI, fake),
            default_answer_model="gpt-4o",
            default_grader_model="gpt-4o-mini",
        ),
        completions,
    )


def test_generate_structured_retry_timeout_is_bounded_by_total_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a retry must not get a fresh full per-attempt timeout -
    the total wall-clock time across every attempt plus backoff must stay
    within the requested timeout_seconds budget."""
    clock = _FakeClock()
    monkeypatch.setattr(chat_client.time, "perf_counter", clock.perf_counter)
    monkeypatch.setattr(chat_client.time, "sleep", clock.sleep)
    monkeypatch.setattr(chat_client, "_RETRY_BACKOFF_SECONDS", 2.0)
    llm, fake = _client_with_clock(
        clock,
        [
            (6.0, _rate_limit_error("slow failure")),  # eats 6s of a 10s budget
            (0.0, _fake_completion('{"value": "recovered"}')),
        ],
    )

    result = llm.generate_structured("system", "user", response_model=_Answer, timeout_seconds=10.0)

    assert result.value.value == "recovered"
    assert len(fake.calls) == 2
    assert fake.calls[0]["timeout"] == 10.0
    # Second attempt's timeout is clamped to what's actually left (10 - 6
    # elapsed - 2 backoff = 2), not the original 10s budget again.
    assert fake.calls[1]["timeout"] == pytest.approx(2.0)
    assert clock.sleeps == [pytest.approx(2.0)]


def test_generate_structured_gives_up_without_retrying_past_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: if a failed attempt already consumed the whole budget,
    the client must raise immediately - not sleep for backoff and burn
    another HTTP call it has no time left for."""
    clock = _FakeClock()
    monkeypatch.setattr(chat_client.time, "perf_counter", clock.perf_counter)
    monkeypatch.setattr(chat_client.time, "sleep", clock.sleep)
    monkeypatch.setattr(chat_client, "_RETRY_BACKOFF_SECONDS", 2.0)
    llm, fake = _client_with_clock(
        clock,
        [
            (5.0, _rate_limit_error("used the whole budget")),
            (0.0, _fake_completion("never reached")),
        ],
    )

    with pytest.raises(RateLimitError, match="used the whole budget"):
        llm.generate_structured("system", "user", response_model=_Answer, timeout_seconds=5.0)

    assert len(fake.calls) == 1
    assert clock.sleeps == []


def test_generate_structured_schema_makes_optional_fields_required_and_nullable() -> None:
    """Regression: OpenAI's strict Structured Outputs mode rejects a schema
    where an optional/defaulted field is missing from 'required' (confirmed
    live: "'required' is required to be supplied and to be an array
    including every key in properties") - every generate_structured call
    must patch this, not just models a developer remembered to hand-craft
    around it."""
    llm, fake = _client(_fake_completion('{"value": "x", "note": null}'))

    llm.generate_structured("system", "user", response_model=_AnswerWithOptionalField)

    schema = fake.completions.calls[0]["response_format"]["json_schema"]["schema"]
    assert schema["required"] == ["value", "note"]
    assert schema["additionalProperties"] is False
    assert "default" not in schema["properties"]["note"]


def test_invalid_max_attempts_is_rejected_before_any_call() -> None:
    llm, fake = _client(_fake_completion("unused"))

    with pytest.raises(ValueError, match="max_attempts"):
        llm.generate_structured("system", "user", response_model=_Answer, max_attempts=0)

    assert fake.completions.calls == []
