"""Real-time shock detector.

Consumes a live trade stream (one stream multiplexed across many markets),
maintains a per-market sliding window and cooldown, runs the shock detector on
every incoming trade, and -- when a shock fires -- classifies it, looks up the
historical depth distribution and emits a fully-formed :class:`ShockEvent` to an
async callback (typically the execution engine).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

import config
from classifier.bucket import bucket_key
from classifier.shock import detect_shock
from data.models import ShockEvent, Trade
from detector.window import SlidingWindow
from distributions import store

Level = tuple[float, float]


@dataclass
class MarketContext:
    """Live match context for a market, used to classify shocks.

    In production this is populated from a score/order-book feed. ``kickoff_ms``
    anchors match elapsed time; ``bids`` is the current order book; ``goal_diff``
    is the current absolute goal difference for the priced team.
    """

    slug: str = ""
    kickoff_ms: Optional[int] = None
    goal_diff: int = 0
    bids: list[Level] = field(default_factory=list)


# Provider returns the live context for a market id.
ContextProvider = Callable[[str], MarketContext]

OnShock = Callable[[ShockEvent], Awaitable[None]]


class LiveDetector:
    def __init__(
        self,
        distributions: dict,
        on_shock: OnShock,
        *,
        context_provider: Optional[ContextProvider] = None,
        dome_client=None,
    ) -> None:
        self.distributions = distributions
        self.on_shock = on_shock
        self.context_provider = context_provider
        self.dome_client = dome_client
        self.windows: dict[str, SlidingWindow] = {}
        self.cooldowns: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Core trade handler (directly unit-testable, no network required)
    # ------------------------------------------------------------------
    async def feed_trade(self, trade: Trade) -> Optional[ShockEvent]:
        """Process a single trade; emit a ShockEvent if a shock fires."""

        window = self.windows.get(trade.market_id)
        if window is None:
            window = SlidingWindow()
            self.windows[trade.market_id] = window
        window.add(trade)

        shock = detect_shock(window.trades, self.cooldowns)
        if shock is None:
            return None

        ctx = (
            self.context_provider(trade.market_id)
            if self.context_provider
            else MarketContext(slug=trade.slug)
        )
        kickoff = ctx.kickoff_ms if ctx.kickoff_ms is not None else window.trades[0].timestamp_ms
        elapsed_ms = max(0, shock["detected_at_ms"] - kickoff)
        slug = ctx.slug or trade.slug
        pre_price = shock["peak"]

        key = bucket_key(slug, pre_price, ctx.bids, elapsed_ms, ctx.goal_diff)

        event = ShockEvent(
            market_id=trade.market_id,
            slug=slug,
            peak=shock["peak"],
            floor=shock["floor"],
            depth=shock["depth"],
            pre_price=pre_price,
            elapsed_ms=elapsed_ms,
            goal_diff=ctx.goal_diff,
            bids=list(ctx.bids),
            bucket_key=key,
            detected_at_ms=shock["detected_at_ms"],
        )

        # Attach the resolved percentile distribution for convenience.
        percentiles = store.lookup(key, self.distributions)
        logger.info(
            "SHOCK {} bucket={} depth={:.3f} percentiles={}",
            trade.market_id,
            key,
            shock["depth"],
            percentiles,
        )

        await self.on_shock(event)
        return event

    # ------------------------------------------------------------------
    # Live run loop
    # ------------------------------------------------------------------
    async def run(
        self,
        market_slugs: Iterable[str],
        *,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        """Subscribe to the Dome order stream and process trades in real time.

        Subscription is by ``market_slugs`` (Dome's WS filters are users /
        condition_ids / market_slugs). Incoming events carry a ``token_id`` which
        becomes ``Trade.market_id``, so per-outcome windows are tracked correctly.
        """

        if self.dome_client is None:
            raise RuntimeError("LiveDetector.run requires a dome_client")

        async def _cb(trade: Trade) -> None:
            await self.feed_trade(trade)

        await self.dome_client.stream_trades(
            market_slugs, _cb, stop_event=stop_event
        )
