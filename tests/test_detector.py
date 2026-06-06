"""Tests for the live shock detector and sliding window."""

from __future__ import annotations

import pytest

from data.models import Trade
from detector.live import LiveDetector, MarketContext
from detector.window import SlidingWindow


def _trade(market_id, ts_ms, price, slug="epl-game"):
    return Trade(market_id=market_id, slug=slug, timestamp_ms=ts_ms, price=price, size=10)


def test_window_eviction():
    w = SlidingWindow()
    w.add(_trade("m1", 0, 0.5))
    w.add(_trade("m1", 121_000, 0.4))  # 121s later, evicts the t=0 trade
    assert len(w) == 1
    assert w.trades[0].timestamp_ms == 121_000


def test_window_keeps_within_window():
    w = SlidingWindow()
    w.add(_trade("m1", 0, 0.5))
    w.add(_trade("m1", 60_000, 0.45))
    w.add(_trade("m1", 119_000, 0.4))
    assert len(w) == 3


@pytest.mark.asyncio
async def test_live_detector_emits_shock():
    fired = []

    async def on_shock(event):
        fired.append(event)

    det = LiveDetector({}, on_shock)
    # Feed a crash 0.50 -> 0.33 within the window.
    await det.feed_trade(_trade("m1", 0, 0.50))
    await det.feed_trade(_trade("m1", 10_000, 0.45))
    event = await det.feed_trade(_trade("m1", 20_000, 0.33))

    assert event is not None
    assert len(fired) == 1
    assert fired[0].market_id == "m1"
    assert fired[0].peak == 0.50
    assert fired[0].floor == 0.33
    assert fired[0].bucket_key.startswith("deep|")  # epl slug


@pytest.mark.asyncio
async def test_live_detector_cooldown():
    fired = []

    async def on_shock(event):
        fired.append(event)

    det = LiveDetector({}, on_shock)
    # First crash.
    await det.feed_trade(_trade("m1", 0, 0.50))
    await det.feed_trade(_trade("m1", 20_000, 0.33))
    # Second crash 60s later, within the 180s cooldown -> blocked.
    await det.feed_trade(_trade("m1", 80_000, 0.50))
    await det.feed_trade(_trade("m1", 90_000, 0.33))

    assert len(fired) == 1


@pytest.mark.asyncio
async def test_multi_market_isolation():
    fired = []

    async def on_shock(event):
        fired.append(event)

    det = LiveDetector({}, on_shock)
    # Two markets crash independently; each should fire once.
    await det.feed_trade(_trade("m1", 0, 0.50))
    await det.feed_trade(_trade("m2", 0, 0.60))
    await det.feed_trade(_trade("m1", 20_000, 0.33))
    await det.feed_trade(_trade("m2", 20_000, 0.40))

    markets = {e.market_id for e in fired}
    assert markets == {"m1", "m2"}
    assert len(fired) == 2


@pytest.mark.asyncio
async def test_context_provider_used():
    fired = []

    async def on_shock(event):
        fired.append(event)

    def provider(market_id):
        return MarketContext(
            slug="epl-game",
            kickoff_ms=0,
            goal_diff=0,
            bids=[(0.5, 20), (0.49, 20), (0.48, 20), (0.47, 20), (0.46, 20)],
        )

    det = LiveDetector({}, on_shock, context_provider=provider)
    await det.feed_trade(_trade("m1", 0, 0.50))
    await det.feed_trade(_trade("m1", 20_000, 0.33))

    assert len(fired) == 1
    # 0.50 peak -> balanced; balanced book; early; level.
    assert fired[0].bucket_key == "deep|balanced|balanced|early|level"
