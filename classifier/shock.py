"""Core shock detection algorithm.

A shock fires when, inside a sliding two-minute window, the price drops at least
15% from the window's peak to its floor, the absolute drop is at least 8 cents,
and the market is not within a 3-minute cooldown of a previous shock.

The 8-cent absolute minimum suppresses noise at very low prices; the cooldown
prevents the same real-world event (a goal, a red card) from being counted
multiple times.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import config
from data.models import Trade


def detect_shock(
    window_trades: Sequence[Trade],
    cooldown_tracker: dict[str, int],
    *,
    drop_pct: float = config.SHOCK_DROP_PCT,
    drop_abs: float = config.SHOCK_DROP_ABS,
    cooldown_ms: int = config.COOLDOWN_MS,
) -> Optional[dict]:
    """Return a shock descriptor dict, or ``None`` if no shock fired.

    ``window_trades`` must be the trades within the last ``SHOCK_WINDOW_MS`` for a
    *single* market, in chronological order. ``cooldown_tracker`` maps
    ``market_id -> last_shock_timestamp_ms`` and is mutated in place when a shock
    fires so the per-market cooldown is enforced across calls.
    """

    if len(window_trades) < 2:
        return None

    market_id = window_trades[0].market_id
    prices = [t.price for t in window_trades]
    peak = max(prices)
    floor = min(prices)
    if peak <= 0:
        return None

    drop_pct_actual = (peak - floor) / peak
    drop_abs_actual = peak - floor
    now_ms = window_trades[-1].timestamp_ms

    last_shock = cooldown_tracker.get(market_id, None)
    if last_shock is not None and (now_ms - last_shock) < cooldown_ms:
        return None  # cooldown active

    if drop_pct_actual >= drop_pct and drop_abs_actual >= drop_abs:
        cooldown_tracker[market_id] = now_ms
        return {
            "shock": True,
            "market_id": market_id,
            "peak": peak,
            "floor": floor,
            "depth": drop_abs_actual,
            "drop_pct": drop_pct_actual,
            "detected_at_ms": now_ms,
        }
    return None
