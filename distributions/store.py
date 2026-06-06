"""Persist and look up the bucket -> percentile distribution table."""

from __future__ import annotations

import json
from typing import Any

import config


def save(distributions: dict[str, Any], path: str | None = None) -> None:
    """Serialize the distribution table to JSON."""

    path = path or config.DISTRIBUTIONS_PATH
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(distributions, fh, indent=2, sort_keys=True)


def load(path: str | None = None) -> dict[str, Any]:
    """Load the distribution table from JSON, validating the percentile fields."""

    path = path or config.DISTRIBUTIONS_PATH
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError("distributions file must contain a JSON object")

    required = {str(p) for p in config.PERCENTILES}
    for key, entry in data.items():
        pct = entry.get("percentiles") if isinstance(entry, dict) else None
        if not isinstance(pct, dict) or not required <= set(pct.keys()):
            raise ValueError(
                f"bucket '{key}' is missing required percentiles {sorted(required)}"
            )
    return data


def _fallback() -> dict[int, float]:
    return dict(config.FALLBACK_PERCENTILES)


def lookup(bucket_key: str, distributions: dict[str, Any]) -> dict[int, float]:
    """Return percentile depths (int-keyed) for a bucket.

    Falls back to conservative shallow defaults when the bucket is missing or its
    distribution is untrusted (too few samples).
    """

    entry = distributions.get(bucket_key)
    if not isinstance(entry, dict):
        return _fallback()
    if entry.get("trusted") is False:
        return _fallback()
    pct = entry.get("percentiles")
    if not isinstance(pct, dict):
        return _fallback()
    try:
        return {int(k): float(v) for k, v in pct.items()}
    except (TypeError, ValueError):
        return _fallback()
