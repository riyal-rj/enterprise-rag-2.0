"""Adapter around the OpenAI chat completions API.

Wraps the OpenAI SDK behind an in-house ``LLMClient`` Protocol, consistent
with every other external service in this codebase (``BcryptPasswordHasher``
wraps bcrypt, ``JWTTokenIssuer`` wraps pyjwt, ``UpstashCacheBackend`` wraps
upstash_redis) — callers depend on the Protocol, never on ``openai.OpenAI``
directly, so the provider can be swapped or faked in tests.

The one real external call this makes (chat completion) runs behind a
retry wrapper, the same treatment ``app.eval.ragas_adapter`` gives its RAGAS
call — LLM APIs are exactly the kind of rate-limited, occasionally-flaky
call that benefits from bounded retries rather than a bare call. (No
separate thread-based timeout wrapper is needed here, unlike
``ragas_adapter``: the OpenAI SDK takes a native ``timeout`` argument.)
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

from openai import OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 2.0
_REQUEST_TIMEOUT_SECONDS = 60.0


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    text: str
    usage: TokenUsage


class LLMClient(Protocol):
    """Contract for a chat-completion provider."""

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse: ...

    def generate_json(
        self,
        system_prompt: str,
        user_message: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


def build_openai_client(api_key: str, timeout_seconds: float = _REQUEST_TIMEOUT_SECONDS) -> OpenAI:
    if not api_key:
        raise ValueError("OpenAI API key must not be empty")
    return OpenAI(api_key=api_key, timeout=timeout_seconds)


class OpenAILLMClient:
    """:class:`LLMClient` backed by the OpenAI SDK.

    ``default_answer_model``/``default_grader_model`` mirror
    ``LLMSettings.llm_model_answer``/``llm_model_grader`` — ``generate``
    (free-text answers, HyDE drafts, ...) defaults to the answer model;
    ``generate_json`` (graders, routers, structured extraction) defaults to
    the cheaper grader model. Either can be overridden per call via
    ``model=``.
    """

    def __init__(
        self, client: OpenAI, default_answer_model: str, default_grader_model: str
    ) -> None:
        self._client = client
        self._default_answer_model = default_answer_model
        self._default_grader_model = default_grader_model

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        return self._complete(
            system_prompt,
            user_message,
            model or self._default_answer_model,
            temperature,
            response_format=None,
        )

    def generate_json(
        self,
        system_prompt: str,
        user_message: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        return self._complete(
            system_prompt,
            user_message,
            model or self._default_grader_model,
            temperature,
            response_format={"type": "json_object"},
        )

    def _complete(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
        temperature: float,
        response_format: dict[str, str] | None,
    ) -> LLMResponse:
        kwargs: dict[str, object] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = self._create_with_retries(model, kwargs)

        text = response.choices[0].message.content or ""
        usage = response.usage
        return LLMResponse(
            text=text,
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
        )

    def _create_with_retries(self, model: str, kwargs: dict[str, object]) -> ChatCompletion:
        last_error: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                # kwargs is built dynamically (with/without response_format);
                # OpenAI's heavily-overloaded signature can't be statically
                # matched against a dict, even though the shape is valid.
                return self._client.chat.completions.create(**kwargs)  # type: ignore[call-overload]
            except Exception as exc:  # noqa: BLE001 - retry any transient API failure
                last_error = exc
                logger.warning(
                    "llm.completion_retry",
                    extra={
                        "model": model,
                        "attempt": attempt,
                        "max_attempts": _MAX_ATTEMPTS,
                        "error": str(exc),
                    },
                )
                if attempt < _MAX_ATTEMPTS:
                    time.sleep(_RETRY_BACKOFF_SECONDS * attempt)

        assert last_error is not None  # loop always sets it before exhausting attempts
        raise last_error
