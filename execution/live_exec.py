"""Live execution against the Polymarket CLOB.

Wraps ``py-clob-client`` to place the laddered limit buy orders, cancel unfilled
orders after their TTL, and exit filled positions at the recovery target.

This module is intentionally defensive: the CLOB client is only imported and
constructed when live execution is actually used, so paper-mode runs and the
test suite never need real credentials or network access.

IMPORTANT: live trading risks real capital. Only enable after backtesting shows
positive expectancy, and start small.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from loguru import logger

import config
from data.models import ShockEvent


class LiveExecution:
    def __init__(
        self,
        storage=None,
        *,
        host: str = "https://clob.polymarket.com",
        chain_id: int = 137,
        client: Any | None = None,
    ) -> None:
        self.storage = storage
        self.host = host
        self.chain_id = chain_id
        self._client = client
        self.open_orders: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Lazy client construction
    # ------------------------------------------------------------------
    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> Any:  # pragma: no cover - requires network/creds
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds

        if not (
            config.POLYMARKET_API_KEY
            and config.POLYMARKET_API_SECRET
            and config.POLYMARKET_API_PASSPHRASE
        ):
            raise RuntimeError(
                "Live execution requires POLYMARKET_API_KEY / _SECRET / _PASSPHRASE"
            )
        creds = ApiCreds(
            api_key=config.POLYMARKET_API_KEY,
            api_secret=config.POLYMARKET_API_SECRET,
            api_passphrase=config.POLYMARKET_API_PASSPHRASE,
        )
        client = ClobClient(self.host, chain_id=self.chain_id, creds=creds)
        return client

    # ------------------------------------------------------------------
    # Submit ladder
    # ------------------------------------------------------------------
    def submit(self, shock: ShockEvent, orders: list[dict]) -> list[dict]:
        """Place GTC limit buy orders for the ladder and record them."""

        placed = []
        for o in orders:
            shares = o["size"] / o["price"] if o["price"] else 0.0
            order_id = self._place_limit_buy(shock.market_id, o["price"], shares)
            record = {
                "order_id": order_id,
                "market_id": shock.market_id,
                "bucket_key": shock.bucket_key,
                "percentile": o["percentile"],
                "price": o["price"],
                "size": o["size"],
                "status": "open",
                "fill_price": None,
                "exit_price": None,
                "pnl": None,
                "created_at_ms": shock.detected_at_ms,
                "updated_at_ms": int(time.time() * 1000),
            }
            self.open_orders[order_id] = record
            placed.append(record)
            if self.storage is not None:
                self.storage.insert_order(record)
        return placed

    def _place_limit_buy(self, market_id: str, price: float, shares: float) -> str:  # pragma: no cover
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY

        args = OrderArgs(
            token_id=market_id,
            price=round(price, 2),
            size=round(shares, 2),
            side=BUY,
        )
        signed = self.client.create_order(args)
        resp = self.client.post_order(signed, OrderType.GTC)
        order_id = resp.get("orderID") or resp.get("order_id") or ""
        logger.info("Placed live buy {} @ {} x {}", order_id, price, shares)
        return order_id

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def cancel_expired(self, now_ms: Optional[int] = None) -> list[str]:  # pragma: no cover
        now_ms = now_ms or int(time.time() * 1000)
        ttl_ms = config.ORDER_TTL_SECONDS * 1000
        cancelled = []
        for order_id, rec in list(self.open_orders.items()):
            if rec["status"] == "open" and now_ms - rec["created_at_ms"] > ttl_ms:
                self.client.cancel(order_id)
                rec["status"] = "cancelled"
                cancelled.append(order_id)
        return cancelled

    def check_fills(self) -> None:  # pragma: no cover - requires network
        for order_id, rec in list(self.open_orders.items()):
            if rec["status"] != "open":
                continue
            status = self.client.get_order(order_id)
            if status and status.get("status") == "matched":
                rec["status"] = "filled"
                rec["fill_price"] = rec["price"]
                # Record the stop-loss threshold for this position.
                if config.STOP_LOSS_CENTS is not None:
                    rec["stop_price"] = round(rec["fill_price"] - config.STOP_LOSS_CENTS, 2)
                self._place_exit(rec)

    def check_stops(self, market_id: str, current_price: float) -> None:  # pragma: no cover
        """Cut filled positions whose price has breached the stop loss.

        Cancels the resting take-profit sell and dumps the position at market.
        No-op when ``STOP_LOSS_CENTS`` is disabled.
        """

        if config.STOP_LOSS_CENTS is None:
            return
        for rec in list(self.open_orders.values()):
            if rec["status"] != "filled" or rec["market_id"] != market_id:
                continue
            stop = rec.get("stop_price")
            if stop is not None and current_price <= stop:
                exit_id = rec.get("exit_order_id")
                if exit_id:
                    try:
                        self.client.cancel(exit_id)
                    except Exception as exc:  # best-effort
                        logger.warning("Failed to cancel TP {}: {}", exit_id, exc)
                self._place_market_sell(rec, current_price)
                rec["status"] = "stopped"
                rec["exit_price"] = current_price
                logger.info("Stop-loss hit for {} @ {}", rec["order_id"], current_price)

    def _place_exit(self, rec: dict) -> None:  # pragma: no cover - requires network
        """Place the take-profit limit sell at fill_price + EXIT_CENTS."""

        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import SELL

        target = rec["fill_price"] + config.EXIT_CENTS
        shares = rec["size"] / rec["fill_price"]
        args = OrderArgs(
            token_id=rec["market_id"], price=round(target, 2), size=round(shares, 2), side=SELL
        )
        signed = self.client.create_order(args)
        resp = self.client.post_order(signed, OrderType.GTC)
        rec["exit_order_id"] = resp.get("orderID") or resp.get("order_id") or ""
        logger.info("Placed take-profit for {} @ {}", rec["order_id"], target)

    def _place_market_sell(self, rec: dict, price: float) -> None:  # pragma: no cover
        """Aggressively sell the position to realize the stop loss."""

        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import SELL

        shares = rec["size"] / rec["fill_price"]
        args = OrderArgs(
            token_id=rec["market_id"],
            price=round(max(price, config.MIN_ORDER_PRICE), 2),
            size=round(shares, 2),
            side=SELL,
        )
        signed = self.client.create_order(args)
        self.client.post_order(signed, OrderType.GTC)
