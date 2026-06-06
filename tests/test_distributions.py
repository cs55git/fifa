"""Tests for the distribution builder and store."""

from __future__ import annotations

import config
from distributions import store
from distributions.builder import (
    DistributionBuilder,
    ShockContext,
    compute_percentiles,
)
from data.models import Trade


def test_compute_percentiles_example():
    # Article example sorted depths (in cents); compare in cents for clarity.
    depths = [4, 5, 6, 7, 8, 10, 12, 15, 20, 25]
    pct = compute_percentiles(depths)
    assert abs(pct[50] - 9.0) < 0.01
    assert abs(pct[75] - 14.25) < 0.01  # linear interpolation: 12 + 0.75*(15-12)
    assert abs(pct[90] - 20.5) < 0.01  # 20 + 0.1*(25-20)
    assert pct[95] > pct[90]


def test_compute_percentiles_monotonic():
    pct = compute_percentiles([0.04, 0.05, 0.06, 0.08, 0.20])
    assert pct[50] <= pct[75] <= pct[90] <= pct[95]


def test_thin_bucket_fallback():
    builder = DistributionBuilder()
    for d in (0.05, 0.06, 0.07):  # only 3 samples -> untrusted
        builder.add_depth("deep|underdog|balanced|mid|level", d)
    dists = builder.build()
    entry = dists["deep|underdog|balanced|mid|level"]
    assert entry["trusted"] is False
    assert entry["count"] == 3
    looked = store.lookup("deep|underdog|balanced|mid|level", dists)
    assert looked == dict(config.FALLBACK_PERCENTILES)


def test_trusted_bucket_uses_real_percentiles():
    builder = DistributionBuilder()
    for d in (0.04, 0.05, 0.06, 0.08, 0.10, 0.20):  # 6 samples -> trusted
        builder.add_depth("deep|moderate_fav|balanced|mid|level", d)
    dists = builder.build()
    entry = dists["deep|moderate_fav|balanced|mid|level"]
    assert entry["trusted"] is True
    looked = store.lookup("deep|moderate_fav|balanced|mid|level", dists)
    assert looked != dict(config.FALLBACK_PERCENTILES)
    assert looked[95] >= looked[50]


def test_save_load_roundtrip(tmp_path):
    builder = DistributionBuilder()
    for d in (0.04, 0.05, 0.06, 0.08, 0.10, 0.20):
        builder.add_depth("deep|moderate_fav|balanced|mid|level", d)
    dists = builder.build()
    path = tmp_path / "distributions.json"
    store.save(dists, str(path))
    reloaded = store.load(str(path))
    assert reloaded == dists


def test_lookup_missing_key_returns_fallback():
    assert store.lookup("nonexistent|key|x|y|z", {}) == dict(config.FALLBACK_PERCENTILES)


def test_load_rejects_malformed(tmp_path):
    import json

    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"k": {"percentiles": {"50": 0.06}}}))
    try:
        store.load(str(path))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_builder_integration_detects_shock():
    # 20 trades forming a clear crash from 0.50 to 0.30 within the window.
    prices = [0.50] * 5 + [0.48, 0.45, 0.42, 0.38, 0.30] + [0.31] * 10
    trades = [
        Trade(
            market_id="m1",
            slug="epl-game",
            timestamp_ms=i * 5_000,
            price=p,
            size=10,
        )
        for i, p in enumerate(prices)
    ]

    builder = DistributionBuilder()

    def ctx(shock, window):
        return ShockContext(slug="epl-game", bids=[(0.5, 20), (0.49, 20), (0.48, 20)], goal_diff=0)

    n = builder.process_market_trades(trades, market_start_ms=0, context_fn=ctx)
    assert n >= 1
    # The detector fires at the first threshold crossing (>= 8c absolute drop),
    # so at least one depth >= 0.08 is recorded under a deep underdog bucket.
    all_depths = [d for depths in builder.buckets.values() for d in depths]
    assert all_depths
    assert max(all_depths) >= 0.08
    # Peak (pre-shock price) is 0.50 -> balanced; 3 bid levels -> top_heavy;
    # detected within first 15 min -> early; tied score -> level.
    assert "deep|balanced|top_heavy|early|level" in builder.buckets
