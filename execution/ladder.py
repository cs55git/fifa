"""Laddered limit-order construction.

Instead of one buy order, the strategy places four laddered limit buys at
increasing depth (lower price) with increasing capital weight toward the deeper,
better-priced levels. Shallow orders fill often but capture less recovery; deep
orders fill rarely but capture the largest recovery, so they get the most
capital.

Each limit price is the pre-shock price minus the corresponding percentile
depth. Levels whose computed price is at or below ``MIN_ORDER_PRICE`` are skipped.
"""

from __future__ import annotations

from collections.abc import Mapping

import config


def build_ladder(
    pre_price: float,
    percentiles: Mapping[int, float],
    capital: float,
    *,
    weights: Mapping[int, float] | None = None,
) -> list[dict]:
    """Build the laddered limit buy orders for a shock.

    ``percentiles`` maps percentile -> depth (price units). ``capital`` is the
    total dollar budget for this shock. Returns a list of order dicts with
    ``percentile``, ``price`` and ``size`` keys, ordered shallow -> deep.
    """

    weights = weights or config.LADDER_WEIGHTS
    orders: list[dict] = []
    for pct in sorted(weights.keys()):
        weight = weights[pct]
        if weight <= 0:
            # Zero/negative weight means this level is disabled. Skip it rather
            # than emit a size-0 order, which would otherwise "fill" with 0 shares
            # and "exit" at 0 PnL, polluting fill/exit/win-rate stats.
            continue
        depth = percentiles.get(pct)
        if depth is None:
            continue
        limit_price = round(pre_price - depth, 2)
        if limit_price <= config.MIN_ORDER_PRICE:
            # Too deep to be a valid order; skip this level.
            continue
        size = round(capital * weight, 4)
        if size <= 0:
            continue
        orders.append({"percentile": pct, "price": limit_price, "size": size})
    return orders
