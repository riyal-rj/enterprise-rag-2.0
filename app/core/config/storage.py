"""Object storage backend configuration for cache spillover."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.core.config.base import EnvBaseSettings

StorageBackend = Literal["local", "s3"]


class StorageSettings(EnvBaseSettings):
    """Selects and configures the cache storage backend.

    ``storage_backend`` is the Strategy key a storage factory dispatches on
    (see ``app.infra.storage``); typing it as a ``Literal`` makes an invalid
    backend name fail at config-load time instead of at first use.
    """

    storage_backend: StorageBackend = Field(default="local")
    s3_cache_bucket: str = Field(default="adv-rag-cache")
    aws_region: str = Field(default="us-east-1")
