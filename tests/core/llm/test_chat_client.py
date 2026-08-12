from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
from openai import OpenAI

from app.core.llm import chat_client
from app.core.llm.chat_client import OpenAILLMClient, build_openai_client


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
    llm, fake = _client(RuntimeError("transient"), _fake_completion("recovered"))

    result = llm.generate("system", "user")

    assert result.text == "recovered"
    assert len(fake.completions.calls) == 2


def test_raises_after_exhausting_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_client, "_RETRY_BACKOFF_SECONDS", 0.001)
    llm, fake = _client(
        RuntimeError("fail 1"), RuntimeError("fail 2"), RuntimeError("permanent failure")
    )

    with pytest.raises(RuntimeError, match="permanent failure"):
        llm.generate("system", "user")

    assert len(fake.completions.calls) == 3
