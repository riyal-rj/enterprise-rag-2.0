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
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 2.0
_REQUEST_TIMEOUT_SECONDS = 60.0

# Only these are worth a retry: connection drops, timeouts, rate limits,
# and 5xx (the server's problem, plausibly transient). Everything else -
# bad auth, no permission, a malformed request, an unsupported
# schema/model - will fail identically on every attempt, so retrying it
# just burns _MAX_ATTEMPTS * _RETRY_BACKOFF_SECONDS of latency before
# surfacing the same error the first attempt already had.
_TRANSIENT_ERROR_TYPES: tuple[type[Exception], ...] = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)

# Explicit allowlist for retry, not an allowlist for immediate failure: an
# exception type we haven't classified either way also fails fast (see
# _is_transient) rather than being silently retried - a bug in our own
# code shouldn't get 3 attempts to reproduce itself before surfacing.
_PERMANENT_ERROR_TYPES: tuple[type[Exception], ...] = (
    AuthenticationError,
    PermissionDeniedError,
    BadRequestError,
    NotFoundError,
    UnprocessableEntityError,
)


def _is_transient(exc: Exception) -> bool:
    return isinstance(exc, _TRANSIENT_ERROR_TYPES) and not isinstance(exc, _PERMANENT_ERROR_TYPES)


StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    text: str
    usage: TokenUsage


@dataclass(frozen=True)
class StructuredLLMResponse(Generic[StructuredModel]):  # noqa: UP046 - PEP 695 class
    # type params are Python 3.12+ only; this venv (and the pinned lockfile)
    # run 3.11, where `class Foo[T]:` is a SyntaxError, not just a style
    # nit - ruff's target-version = "py312" is aspirational here, not the
    # actual interpreter. Confirmed via `python -c "import ...chat_client"`
    # raising SyntaxError before this suppression was added.
    value: StructuredModel
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

    def generate_structured(
        self,
        system_prompt: str,
        user_message: str,
        *,
        response_model: type[StructuredModel],
        model: str | None = None,
        temperature: float = 0.0,
        max_completion_tokens: int = 1_000,
        timeout_seconds: float = 30.0,
        max_attempts: int = 2,
    ) -> StructuredLLMResponse[StructuredModel]: ...


def build_openai_client(api_key: str, timeout_seconds: float = _REQUEST_TIMEOUT_SECONDS) -> OpenAI:
    # max_retries=0: the SDK's own retry loop is disabled deliberately -
    # _create_with_retries below is the one place that retries, so it can
    # apply this module's transient/permanent classification instead of the
    # SDK's blunter "retry on 429/5xx" default, and so a permanent error
    # fails after exactly one real HTTP call, not the SDK's retries nested
    # inside our own.
    if not api_key:
        raise ValueError("OpenAI API key must not be empty")
    return OpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=0)


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

    def generate_structured(
        self,
        system_prompt: str,
        user_message: str,
        *,
        response_model: type[StructuredModel],
        model: str | None = None,
        temperature: float = 0.0,
        max_completion_tokens: int = 1_000,
        timeout_seconds: float = 30.0,
        max_attempts: int = 2,
    ) -> StructuredLLMResponse[StructuredModel]:
        resolved_model = model or self._default_grader_model
        schema_name = response_model.__name__.lower()
        kwargs: dict[str, object] = {
            "model": resolved_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "timeout": timeout_seconds,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": response_model.model_json_schema(),
                },
            },
        }

        response = self._create_with_retries(
            resolved_model,
            kwargs,
            max_attempts=max_attempts,
        )

        message = response.choices[0].message
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise RuntimeError(f"structured output refused by model {resolved_model}")
        if not message.content:
            raise ValueError("structured output was empty")

        usage = response.usage
        parsed = response_model.model_validate_json(message.content)
        return StructuredLLMResponse(
            value=parsed,
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
        )

    def _create_with_retries(
        self, model: str, kwargs: dict[str, object], max_attempts: int = _MAX_ATTEMPTS
    ) -> ChatCompletion:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")

        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                # kwargs is built dynamically (with/without response_format);
                # OpenAI's heavily-overloaded signature can't be statically
                # matched against a dict, even though the shape is valid.
                return self._client.chat.completions.create(**kwargs)  # type: ignore[call-overload]
            except Exception as exc:  # noqa: BLE001 - classified below; permanent failures re-raise immediately
                last_error = exc
                if not _is_transient(exc):
                    logger.warning(
                        "llm.completion_failed_permanent",
                        extra={
                            "model": model,
                            "attempt": attempt,
                            "error_type": type(exc).__name__,
                        },
                    )
                    raise
                logger.warning(
                    "llm.completion_retry",
                    extra={
                        "model": model,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "error_type": type(exc).__name__,
                    },
                )
                if attempt < max_attempts:
                    time.sleep(_RETRY_BACKOFF_SECONDS * attempt)

        assert last_error is not None  # loop always sets it before exhausting attempts
        raise last_error
