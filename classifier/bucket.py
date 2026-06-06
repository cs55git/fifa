"""Bucket key construction.

Combines the five classification dimensions into a single pipe-delimited key,
e.g. ``"deep|underdog|balanced|mid|level"``. There are 720 theoretical
combinations; in practice roughly 200 accumulate enough data to trade.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from classifier.dimensions import (
    classify_depth,
    classify_favoritism,
    classify_goal,
    classify_league,
    classify_time,
)

Level = tuple[float, float]


def bucket_key(
    slug: str,
    pre_price: float,
    bids: Sequence[Level],
    elapsed_ms: int,
    goal_diff: int,
) -> str:
    """Build the five-dimension bucket key for a shock."""

    return "|".join(
        [
            classify_league(slug),
            classify_favoritism(pre_price),
            classify_depth(bids),
            classify_time(elapsed_ms),
            classify_goal(goal_diff),
        ]
    )


def bucket_matches(key: str, patterns: Optional[Sequence[str]]) -> bool:
    """Whether a bucket key passes a filter of substring patterns.

    The key is ``league|favoritism|depth|time|goal`` (e.g.
    ``"top|moderate_fav|balanced|late|level"``). With no patterns everything
    matches; otherwise the key must contain at least one pattern as a substring,
    so dimension values (``"moderate_fav"``, ``"late"``) or whole keys both work.
    """

    if not patterns:
        return True
    return any(p in key for p in patterns)
