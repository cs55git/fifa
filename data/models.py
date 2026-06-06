"""Pydantic data models shared across the trading system.

A small note on the price/size representation: ``price`` is expressed in
probability units in the range (0, 1) -- i.e. "50 cents" is ``0.50``. The
article frequently speaks in cents; throughout the code we keep everything in
the 0-1 float representation and only format as cents for display.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# An order-book level is a (price, size) pair.
Level = tuple[float, float]


class Trade(BaseModel):
    """A single executed trade for a market outcome token."""

    market_id: str
    slug: str
    timestamp_ms: int
    price: float
    size: float


class OrderBook(BaseModel):
    """A point-in-time snapshot of the order book for a market."""

    market_id: str
    timestamp_ms: int
    bids: list[Level] = Field(default_factory=list)
    asks: list[Level] = Field(default_factory=list)


class ShockEvent(BaseModel):
    """A detected price shock, fully classified and ready for execution.

    ``pre_price`` is the implied probability immediately before the shock
    (the window peak) and is what the ladder is built around. ``depth`` is the
    absolute drop ``peak - floor`` in price units.
    """

    market_id: str
    slug: str
    peak: float
    floor: float
    depth: float
    pre_price: float
    elapsed_ms: int
    goal_diff: int
    bids: list[Level] = Field(default_factory=list)
    bucket_key: str
    detected_at_ms: int
