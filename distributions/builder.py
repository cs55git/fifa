"""Build historical shock-depth distributions per bucket.

Replays historical trades through the same shock detector used live, classifies
every detected shock into its bucket, and records the shock depth (in price
units) under that bucket key. After processing, :meth:`build` converts each
bucket's depth list into a percentile distribution -- falling back to
conservative shallow defaults for buckets with too few samples.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Optional

import numpy as np

import config
from classifier.bucket import bucket_key
from classifier.shock import detect_shock
from data.models import Trade

Level = tuple[float, float]

# Optional context supplied per shock to refine the bucket classification:
# returns (slug, bids, goal_diff). elapsed_ms is computed from match start.
ContextFn = Callable[[dict, Sequence[Trade]], "ShockContext"]


class ShockContext:
    """Side-channel context for classifying a historical shock."""

    def __init__(
        self,
        slug: str = "",
        bids: Optional[Sequence[Level]] = None,
        goal_diff: int = 0,
    ) -> None:
        self.slug = slug
        self.bids = list(bids) if bids else []
        self.goal_diff = goal_diff


def compute_percentiles(depths: Sequence[float]) -> dict[int, float]:
    """Compute P50/P75/P90/P95 of a list of shock depths via linear interpolation."""

    arr = np.sort(np.asarray(depths, dtype=float))
    return {pct: float(np.percentile(arr, pct)) for pct in config.PERCENTILES}


class DistributionBuilder:
    """Accumulates shock depths per bucket and builds percentile distributions."""

    def __init__(self) -> None:
        self.buckets: dict[str, list[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def process_market_trades(
        self,
        trades: Sequence[Trade],
        *,
        market_start_ms: Optional[int] = None,
        context_fn: Optional[ContextFn] = None,
    ) -> int:
        """Replay one market's trades and record any detected shocks.

        ``market_start_ms`` is treated as kickoff; match elapsed time for each
        shock is ``detected_at_ms - market_start_ms``. If omitted, the first
        trade's timestamp is used. ``context_fn`` may supply slug/bids/goal_diff
        for richer classification; without it we use the trade slug, an empty
        book (-> ``unknown`` depth) and a level score.

        Returns the number of shocks recorded.
        """

        if not trades:
            return 0

        ordered = sorted(trades, key=lambda t: t.timestamp_ms)
        if market_start_ms is None:
            market_start_ms = ordered[0].timestamp_ms

        cooldown: dict[str, int] = {}
        window: list[Trade] = []
        count = 0

        for trade in ordered:
            window.append(trade)
            # Evict trades older than the 2-minute window.
            cutoff = trade.timestamp_ms - config.SHOCK_WINDOW_MS
            window = [t for t in window if t.timestamp_ms >= cutoff]

            shock = detect_shock(window, cooldown)
            if shock is None:
                continue

            ctx = context_fn(shock, window) if context_fn else ShockContext(slug=trade.slug)
            elapsed_ms = max(0, shock["detected_at_ms"] - market_start_ms)
            key = bucket_key(
                ctx.slug or trade.slug,
                shock["peak"],  # pre-shock price = window peak
                ctx.bids,
                elapsed_ms,
                ctx.goal_diff,
            )
            self.buckets[key].append(shock["depth"])
            count += 1

        return count

    def add_depth(self, key: str, depth: float) -> None:
        """Directly record a shock depth under a bucket (useful for tooling/tests)."""

        self.buckets[key].append(depth)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self) -> dict[str, dict]:
        """Convert accumulated depths into per-bucket percentile distributions.

        Buckets with fewer than ``MIN_BUCKET_SIZE`` shocks fall back to
        conservative shallow defaults and are flagged ``trusted=False``.
        """

        distributions: dict[str, dict] = {}
        for key, depths in self.buckets.items():
            if len(depths) < config.MIN_BUCKET_SIZE:
                pct = dict(config.FALLBACK_PERCENTILES)
                trusted = False
            else:
                pct = compute_percentiles(depths)
                trusted = True
            distributions[key] = {
                "percentiles": {str(k): v for k, v in pct.items()},
                "count": len(depths),
                "trusted": trusted,
            }
        return distributions
