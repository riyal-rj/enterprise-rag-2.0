"""LLM and embedding provider configuration."""

from __future__ import annotations

from pydantic import Field, SecretStr

from app.core.config.base import EnvBaseSettings


class LLMSettings(EnvBaseSettings):
    """Model selection and credentials for answer/grading LLM calls."""

    openai_api_key: SecretStr = Field(default=SecretStr(""))
    llm_model_answer: str = Field(default="gpt-4o")
    llm_model_grader: str = Field(default="gpt-4o-mini")
    embedding_model: str = Field(default="text-embedding-3-small")
