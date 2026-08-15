"""Document ingestion pipeline configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.core.config.base import EnvBaseSettings

AcceleratorDeviceName = Literal["auto", "cpu", "cuda", "mps", "xpu"]


class IngestionSettings(EnvBaseSettings):
    """Tuning knobs for the Docling document-conversion/chunking pipeline.

    ``accelerator_device`` defaults to ``"auto"`` rather than pinning a
    specific device (e.g. ``"mps"``, Apple-Silicon-only) — the ingestion
    pipeline runs wherever the service is deployed, and a hardcoded device
    would raise on hardware that lacks it. Docling resolves ``"auto"`` to
    the best available device at conversion time.
    """

    accelerator_device: AcceleratorDeviceName = Field(default="auto")
    accelerator_num_threads: int = Field(default=8, gt=0)

    max_upload_size_mb: int = Field(default=25, gt=0)
