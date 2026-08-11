"""Loads and validates golden eval cases from YAML."""

from __future__ import annotations

import warnings
from pathlib import Path

import yaml
from pydantic import TypeAdapter

from app.eval.schemas import EvalFeature, GoldenCase

_DEFAULT_GOLDENS_PATH = Path(__file__).parent / "data" / "goldens.yaml"

_golden_cases_adapter: TypeAdapter[list[GoldenCase]] = TypeAdapter(list[GoldenCase])


def load_golden_cases(path: Path | None = None) -> list[GoldenCase]:
    """Load and validate golden eval cases from a YAML file.

    Raises:
        ValueError: if the YAML root isn't a list, or if any case ``id`` is
            duplicated.
        pydantic.ValidationError: if any case fails schema validation.
    """
    resolved_path = path or _DEFAULT_GOLDENS_PATH
    raw = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected YAML root to be a list, got {type(raw).__name__}")

    cases = _golden_cases_adapter.validate_python(raw)

    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        duplicates = {i for i in ids if ids.count(i) > 1}
        raise ValueError(f"Duplicate golden IDs found: {duplicates}")

    _warn_on_missing_feature_coverage(cases)
    return cases


def filter_by_feature(cases: list[GoldenCase], feature: str | None) -> list[GoldenCase]:
    """Return only cases whose ``demonstrates_feature`` matches ``feature``.

    Returns ``cases`` unchanged when ``feature`` is ``None``.
    """
    if feature is None:
        return cases
    return [case for case in cases if case.demonstrates_feature.value == feature]


def _warn_on_missing_feature_coverage(cases: list[GoldenCase]) -> None:
    """Warn (don't fail) if some feature categories have no entries.

    Not every feature needs eval goldens — e.g. sparse/dense/security are
    demo-only and may be intentionally excluded from the eval-progression
    set.
    """
    present = {case.demonstrates_feature for case in cases}
    missing = set(EvalFeature) - present
    if missing:
        missing_values = sorted(feature.value for feature in missing)
        warnings.warn(f"No golden entries for features: {missing_values} (OK if demo-only)", stacklevel=2)
