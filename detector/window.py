"""A per-market sliding trade window.

Holds the trades observed within the last ``SHOCK_WINDOW_MS`` and evicts older
ones as new trades arrive. Eviction is keyed off the latest trade's timestamp
(event time) rather than wall-clock time so the same logic works for both live
and replayed data.
"""

from __future__ import annotations

from collections import deque

import config
from data.models import Trade


class SlidingWindow:
    def __init__(self, window_ms: int = config.SHOCK_WINDOW_MS) -> None:
        self.window_ms = window_ms
        self._trades: deque[Trade] = deque()

    def add(self, trade: Trade) -> None:
        """Append a trade and evict any trades older than the window."""

        self._trades.append(trade)
        cutoff = trade.timestamp_ms - self.window_ms
        while self._trades and self._trades[0].timestamp_ms < cutoff:
            self._trades.popleft()

    @property
    def trades(self) -> list[Trade]:
        return list(self._trades)

    def __len__(self) -> int:
        return len(self._trades)

    def reset(self) -> None:
        self._trades.clear()
