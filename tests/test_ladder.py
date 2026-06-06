"""Tests for the laddered limit-order builder."""

from __future__ import annotations

from execution.ladder import build_ladder


# Article worked example weights (independent of the tunable config defaults).
_WEIGHTS = {50: 0.10, 75: 0.20, 90: 0.30, 95: 0.40}


def test_build_ladder_example():
    # Article worked example: pre=0.30, depths 8/12/16/20c, $50 capital.
    percentiles = {50: 0.08, 75: 0.12, 90: 0.16, 95: 0.20}
    orders = build_ladder(0.30, percentiles, 50.0, weights=_WEIGHTS)
    assert len(orders) == 4

    by_pct = {o["percentile"]: o for o in orders}
    assert by_pct[50]["price"] == 0.22
    assert by_pct[75]["price"] == 0.18
    assert by_pct[90]["price"] == 0.14
    assert by_pct[95]["price"] == 0.10

    assert by_pct[50]["size"] == 5.0
    assert by_pct[75]["size"] == 10.0
    assert by_pct[90]["size"] == 15.0
    assert by_pct[95]["size"] == 20.0


def test_build_ladder_weighting_increases_with_depth():
    percentiles = {50: 0.05, 75: 0.08, 90: 0.12, 95: 0.18}
    orders = build_ladder(0.50, percentiles, 100.0, weights=_WEIGHTS)
    sizes = [o["size"] for o in orders]
    assert sizes == sorted(sizes)  # deeper levels get more capital


def test_build_ladder_skips_zero_weight_levels():
    # Disabling a level via 0 weight must NOT emit a size-0 order; such orders
    # would "fill" with 0 shares and pollute fill/exit/win-rate stats.
    percentiles = {50: 0.05, 75: 0.08, 90: 0.12, 95: 0.18}
    weights = {50: 0.0, 75: 0.0, 90: 0.25, 95: 0.75}
    orders = build_ladder(0.50, percentiles, 100.0, weights=weights)
    pcts = {o["percentile"] for o in orders}
    assert pcts == {90, 95}
    assert all(o["size"] > 0 for o in orders)


def test_build_ladder_no_negative_or_subpenny_prices():
    # P95 depth exceeds pre-price -> that level is skipped.
    percentiles = {50: 0.05, 75: 0.10, 90: 0.20, 95: 0.40}
    orders = build_ladder(0.30, percentiles, 50.0)
    pcts = {o["percentile"] for o in orders}
    assert 95 not in pcts  # 0.30 - 0.40 < 0
    assert all(o["price"] > 0.01 for o in orders)


def test_build_ladder_skips_at_penny_floor():
    percentiles = {50: 0.29, 75: 0.30, 90: 0.31, 95: 0.32}
    orders = build_ladder(0.30, percentiles, 50.0)
    # 0.30-0.29 = 0.01 which is not > MIN_ORDER_PRICE (0.01) -> skipped.
    assert orders == []
