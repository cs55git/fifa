"""Paper trading engine.

Simulates the laddered execution without touching real capital. Orders are held
in memory (and optionally persisted to storage). Fills, exits and expiries are
driven by subsequent price updates fed in via :meth:`update`.

Economics follow the article: a filled limit buy of ``size`` dollars at
``fill_price`` buys ``size / fill_price`` shares. The position exits when the
price recovers ``EXIT_CENTS`` above the fill, selling the shares back into the
bounce. Realized PnL is ``shares * (exit_price - fill_price)``.
"""

from __future__ import annotations

import itertools
from typing import Any, Optional

from loguru import logger

import config
from data.models import ShockEvent


class PaperEngine:
    def __init__(
        self,
        storage=None,
        *,
        exit_cents: float = config.EXIT_CENTS,
        stop_loss_cents: Optional[float] = config.STOP_LOSS_CENTS,
        max_hold_seconds: Optional[float] = config.MAX_HOLD_SECONDS,
    ) -> None:
        self.storage = storage
        self.exit_cents = exit_cents  # take profit
        self.stop_loss_cents = stop_loss_cents  # cut losses (None disables)
        self.max_hold_seconds = max_hold_seconds  # near-term exit (None disables)
        self.orders: list[dict[str, Any]] = []
        # Active (non-terminal) orders indexed by market so update() only scans
        # the relevant market's orders instead of every order ever submitted.
        self._active: dict[str, list[dict[str, Any]]] = {}
        self._ids = itertools.count(1)

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------
    def submit(self, shock: ShockEvent, orders: list[dict]) -> list[dict]:
        """Register a ladder of orders for a shock as ``open``."""

        created = []
        for o in orders:
            record = {
                "order_id": f"paper-{next(self._ids)}",
                "market_id": shock.market_id,
                "bucket_key": shock.bucket_key,
                "percentile": o["percentile"],
                "price": o["price"],
                "size": o["size"],  # dollar budget for this level
                "status": "open",
                "shares": None,
                "fill_price": None,
                "filled_at_ms": None,
                "exit_price": None,
                "pnl": None,
                "created_at_ms": shock.detected_at_ms,
                "updated_at_ms": shock.detected_at_ms,
            }
            self.orders.append(record)
            self._active.setdefault(shock.market_id, []).append(record)
            created.append(record)
            if self.storage is not None:
                self.storage.insert_order(record)
        logger.info(
            "Submitted {} paper orders for {} ({})",
            len(created),
            shock.market_id,
            shock.bucket_key,
        )
        return created

    # ------------------------------------------------------------------
    # Update (fills / exits / expiries)
    # ------------------------------------------------------------------
    def update(self, market_id: str, current_price: float, current_ts_ms: int) -> None:
        """Advance order state given a new price observation for a market."""

        active = self._active.get(market_id)
        if not active:
            return

        ttl_ms = config.ORDER_TTL_SECONDS * 1000
        still_active: list[dict[str, Any]] = []
        for order in active:
            if order["status"] == "open":
                # Limit buy fills when price touches or crosses the limit.
                if current_price <= order["price"]:
                    order["status"] = "filled"
                    order["fill_price"] = order["price"]
                    order["filled_at_ms"] = current_ts_ms
                    order["shares"] = order["size"] / order["price"]
                    order["updated_at_ms"] = current_ts_ms
                elif current_ts_ms - order["created_at_ms"] > ttl_ms:
                    order["status"] = "expired"
                    order["updated_at_ms"] = current_ts_ms

            elif order["status"] == "filled":
                target = round(order["fill_price"] + self.exit_cents, 6)
                # Take profit: epsilon guards float drift (e.g. 0.14 + 0.04).
                if current_price >= target - 1e-9:
                    order["status"] = "exited"
                    order["exit_price"] = target
                    order["pnl"] = order["shares"] * (target - order["fill_price"])
                    order["updated_at_ms"] = current_ts_ms
                    continue
                # Stop loss: cut the position once price falls the configured
                # distance below the fill. Fills at the next observed price.
                if self.stop_loss_cents is not None:
                    stop = round(order["fill_price"] - self.stop_loss_cents, 6)
                    if current_price <= stop + 1e-9:
                        order["status"] = "stopped"
                        order["exit_price"] = current_price
                        order["pnl"] = order["shares"] * (current_price - order["fill_price"])
                        order["updated_at_ms"] = current_ts_ms
                        continue
                # Time exit: close at market once the position has been held past
                # the max-hold window (models the near-term bounce exit rather
                # than holding to resolution).
                if self.max_hold_seconds is not None:
                    held_ms = current_ts_ms - (order["filled_at_ms"] or order["created_at_ms"])
                    if held_ms >= self.max_hold_seconds * 1000:
                        order["status"] = "timed"
                        order["exit_price"] = current_price
                        order["pnl"] = order["shares"] * (current_price - order["fill_price"])
                        order["updated_at_ms"] = current_ts_ms
                        continue

            # Keep only non-terminal orders active; terminal ones used `continue`
            # above (or are expired) and drop out of the per-market index.
            if order["status"] in ("open", "filled"):
                still_active.append(order)

        self._active[market_id] = still_active

    # ------------------------------------------------------------------
    # Settlement (mark-to-market)
    # ------------------------------------------------------------------
    def settle(self, market_id: str, final_price: float, final_ts_ms: int) -> None:
        """Close out any unresolved orders for a market at a final price.

        A position that never recovered to the +``exit_cents`` target is a real
        outcome too: it is marked to market at ``final_price`` (which for a
        resolved market trends toward 0 or 1), producing a realized win or LOSS.
        Without this step every filled order would eventually look like a winner,
        which is why an un-settled backtest shows a 100% win rate.

        Unfilled ``open`` orders are simply expired (no position taken).
        """

        for order in self._active.get(market_id, []):
            if order["status"] == "filled":
                order["status"] = "settled"
                order["exit_price"] = final_price
                order["pnl"] = order["shares"] * (final_price - order["fill_price"])
                order["updated_at_ms"] = final_ts_ms
            elif order["status"] == "open":
                order["status"] = "expired"
                order["updated_at_ms"] = final_ts_ms
        # Everything for this market is now terminal.
        self._active[market_id] = []

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        # Realized exits include take-profit hits ("exited"), stop-loss cuts
        # ("stopped"), max-hold market exits ("timed") and end-of-data
        # mark-to-market ("settled"); a win is any realized exit with positive PnL.
        realized = {"exited", "stopped", "timed", "settled"}
        total = len(self.orders)
        filled = [o for o in self.orders if o["status"] in {"filled"} | realized]
        exits = [o for o in self.orders if o["status"] in realized]
        wins = [o for o in exits if (o["pnl"] or 0) > 0]
        total_pnl = sum(o["pnl"] or 0.0 for o in exits)

        by_bucket: dict[str, dict[str, Any]] = {}
        for o in self.orders:
            b = by_bucket.setdefault(
                o["bucket_key"],
                {"orders": 0, "filled": 0, "exits": 0, "wins": 0, "pnl": 0.0},
            )
            b["orders"] += 1
            if o["status"] in {"filled"} | realized:
                b["filled"] += 1
            if o["status"] in realized:
                b["exits"] += 1
                b["pnl"] += o["pnl"] or 0.0
                if (o["pnl"] or 0) > 0:
                    b["wins"] += 1

        return {
            "total_orders": total,
            "fill_rate": (len(filled) / total) if total else 0.0,
            "win_rate": (len(wins) / len(exits)) if exits else 0.0,
            "total_pnl": total_pnl,
            "by_bucket": by_bucket,
        }
