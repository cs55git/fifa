"""Client for the Dome prediction-market data API (https://docs.domeapi.io).

Implements the Polymarket REST endpoints and the real-time WebSocket order
stream used by this bot:

- ``GET /polymarket/markets``  -> :meth:`get_markets`
- ``GET /polymarket/orders``   -> :meth:`get_trades` (trade/order history)
- ``GET /polymarket/orderbook``-> :meth:`get_orderbook`
- ``wss://ws.domeapi.io/<key>``-> :meth:`stream_trades`

Conventions taken from the Dome docs:
- REST base URL is ``https://api.domeapi.io/v1`` and requests authenticate with
  an ``x-api-key`` header.
- All REST timestamps are **Unix seconds**; we convert to milliseconds for the
  bot's internal models.
- The tradeable unit is a ``token_id`` (each market has two outcome tokens,
  ``side_a`` and ``side_b``). Internally we map ``Trade.market_id = token_id``
  and ``Trade.slug = market_slug``.
- Pagination is cursor-based via ``pagination.pagination_key`` / ``has_more``.
- The WebSocket key is part of the connection URL: ``wss://ws.domeapi.io/<key>``.

Dome announced EOL on 2026-04-28; if the endpoints are retired, override the
base URLs via ``DOME_BASE_URL`` / ``DOME_WS_URL`` to point at the successor API.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, Optional

import httpx
from loguru import logger

import config
from data.models import OrderBook, Trade

# Status codes worth retrying with backoff.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _to_ms(ts: Any) -> int:
    """Normalize a Dome timestamp (Unix seconds, sometimes ms) to milliseconds."""

    try:
        ts = int(ts)
    except (TypeError, ValueError):
        return 0
    # Seconds-since-epoch values are ~1e9..1e10; scale those to ms. Values that
    # are already in ms (~1e12+) or small synthetic values are left as-is.
    if 1_000_000_000 <= ts < 10_000_000_000:
        ts *= 1000
    return ts


def _to_level(raw: Any) -> tuple[float, float]:
    """Normalize an order-book level into a (price, size) tuple."""

    if isinstance(raw, dict):
        price = raw.get("price", raw.get("p"))
        size = raw.get("size", raw.get("s", raw.get("shares", raw.get("amount"))))
        return float(price), float(size)
    return float(raw[0]), float(raw[1])


class DomeClient:
    """Wrapper around the Dome REST + WebSocket API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        ws_url: str | None = None,
        *,
        client: httpx.Client | None = None,
        max_retries: int = 5,
        backoff_base: float = 0.5,
    ) -> None:
        self.api_key = api_key if api_key is not None else config.DOME_API_KEY
        self.base_url = (base_url or config.DOME_BASE_URL).rstrip("/")
        self.ws_url = (ws_url or config.DOME_WS_URL).rstrip("/")
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        if client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                headers=self._headers(),
                timeout=30.0,
            )
        else:
            # Ensure auth headers are applied even when a client is injected.
            self._client = client
            self._client.headers.update(self._headers())

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            # Dome authenticates REST requests with the x-api-key header.
            headers["x-api-key"] = self.api_key
        return headers

    # ------------------------------------------------------------------
    # Low-level request with exponential backoff on rate limits / 5xx
    # ------------------------------------------------------------------
    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        attempt = 0
        while True:
            response = self._client.request(method, path, **kwargs)
            if response.status_code in _RETRYABLE_STATUS and attempt < self.max_retries:
                delay = self._retry_delay(response, attempt)
                logger.warning(
                    "Dome {} {} -> {}; retrying in {:.2f}s (attempt {}/{})",
                    method,
                    path,
                    response.status_code,
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                self._sleep(delay)
                attempt += 1
                continue
            response.raise_for_status()
            if not response.content:
                return {}
            return response.json()

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return self.backoff_base * (2**attempt)

    def _sleep(self, seconds: float) -> None:  # pragma: no cover - trivial
        import time

        time.sleep(seconds)

    @staticmethod
    def _pagination_key(payload: Any) -> Optional[str]:
        if not isinstance(payload, dict):
            return None
        pag = payload.get("pagination")
        if isinstance(pag, dict):
            if pag.get("has_more") and pag.get("pagination_key"):
                return str(pag["pagination_key"])
        return None

    # ------------------------------------------------------------------
    # Markets  (GET /polymarket/markets)
    # ------------------------------------------------------------------
    def get_markets(
        self,
        *,
        tags: Optional[Iterable[str]] = None,
        search: Optional[str] = None,
        status: Optional[str] = "open",
        min_volume: Optional[float] = None,
        market_slugs: Optional[Iterable[str]] = None,
        event_slugs: Optional[Iterable[str]] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Return Polymarket market metadata, paginated.

        Each returned dict carries at least ``market_slug``, ``condition_id`` and
        a ``tokens`` list of ``{"token_id", "label"}`` (from ``side_a``/``side_b``).
        Filter by ``tags`` (e.g. ``["Soccer"]``), a fuzzy ``search`` string, or
        explicit slugs. ``search`` cannot be combined with other filters.
        """

        markets: list[dict] = []
        pagination_key: Optional[str] = None
        while True:
            params: list[tuple[str, Any]] = [("limit", min(limit, 100))]
            # status is always allowed and is REQUIRED when using search.
            if status:
                params.append(("status", status))
            if search:
                params.append(("search", search))
            else:
                for t in tags or []:
                    params.append(("tags", t))
                for s in market_slugs or []:
                    params.append(("market_slug", s))
                for e in event_slugs or []:
                    params.append(("event_slug", e))
                if min_volume is not None:
                    params.append(("min_volume", min_volume))
            if pagination_key:
                params.append(("pagination_key", pagination_key))

            payload = self._request("GET", "/polymarket/markets", params=params)
            rows = payload.get("markets", []) if isinstance(payload, dict) else []
            for m in rows:
                markets.append(self._normalize_market(m))

            pagination_key = self._pagination_key(payload)
            if not pagination_key or not rows:
                break
        return markets

    @staticmethod
    def _normalize_market(m: dict) -> dict:
        tokens = []
        for side_key in ("side_a", "side_b"):
            side = m.get(side_key)
            if isinstance(side, dict) and side.get("id"):
                tokens.append({"token_id": str(side["id"]), "label": side.get("label", "")})
        return {
            "market_slug": m.get("market_slug", ""),
            "condition_id": m.get("condition_id", ""),
            "title": m.get("title", ""),
            "tags": m.get("tags", []),
            "status": m.get("status"),
            "game_start_time": m.get("game_start_time"),
            "volume_total": m.get("volume_total"),
            "tokens": tokens,
            "raw": m,
        }

    # ------------------------------------------------------------------
    # Trade / order history  (GET /polymarket/orders)
    # ------------------------------------------------------------------
    def get_trades(
        self,
        token_id: Optional[str] = None,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        *,
        market_slug: Optional[str] = None,
        condition_id: Optional[str] = None,
        page_limit: int = 1000,
    ) -> list[Trade]:
        """Fetch executed trades, transparently following cursor pagination.

        Provide exactly one of ``token_id``, ``market_slug`` or ``condition_id``
        (the Dome API rejects combinations). ``start_ts`` / ``end_ts`` are Unix
        seconds. Each order is mapped to a :class:`Trade` with
        ``market_id = token_id``, ``slug = market_slug``, ``size = shares_normalized``
        and ``timestamp_ms`` converted from seconds.
        """

        if sum(x is not None for x in (token_id, market_slug, condition_id)) != 1:
            raise ValueError(
                "provide exactly one of token_id, market_slug or condition_id"
            )

        trades: list[Trade] = []
        pagination_key: Optional[str] = None
        while True:
            params: dict[str, Any] = {"limit": min(page_limit, 1000)}
            if token_id is not None:
                params["token_id"] = token_id
            if market_slug is not None:
                params["market_slug"] = market_slug
            if condition_id is not None:
                params["condition_id"] = condition_id
            if start_ts is not None:
                params["start_time"] = start_ts
            if end_ts is not None:
                params["end_time"] = end_ts
            if pagination_key:
                params["pagination_key"] = pagination_key

            payload = self._request("GET", "/polymarket/orders", params=params)
            rows = payload.get("orders", []) if isinstance(payload, dict) else []
            for row in rows:
                trades.append(self._parse_order(row, fallback_token=token_id))

            pagination_key = self._pagination_key(payload)
            if not pagination_key or not rows:
                break
        return trades

    @staticmethod
    def _parse_order(row: dict, fallback_token: Optional[str] = None) -> Trade:
        token = str(row.get("token_id", fallback_token or ""))
        size = row.get("shares_normalized")
        if size is None:
            size = row.get("shares", 0.0)
        return Trade(
            market_id=token,
            slug=str(row.get("market_slug", "")),
            timestamp_ms=_to_ms(row.get("timestamp", 0)),
            price=float(row.get("price", 0.0)),
            size=float(size or 0.0),
        )

    # ------------------------------------------------------------------
    # Order book snapshot  (GET /polymarket/orderbook)
    # ------------------------------------------------------------------
    def get_orderbook(
        self, token_id: str, at_time: Optional[int] = None
    ) -> OrderBook:
        """Fetch an order-book snapshot for a token (optionally historical)."""

        params: dict[str, Any] = {"token_id": token_id}
        if at_time is not None:
            params["at_time"] = at_time
        payload = self._request("GET", "/polymarket/orderbook", params=params)

        book = payload
        if isinstance(payload, dict):
            # The history endpoint returns snapshots; take the latest if present.
            snaps = payload.get("orderbook") or payload.get("snapshots")
            if isinstance(snaps, list) and snaps:
                book = snaps[-1]
            elif "bids" not in payload and isinstance(payload.get("data"), dict):
                book = payload["data"]

        bids = [_to_level(b) for b in (book.get("bids") or [])]
        asks = [_to_level(a) for a in (book.get("asks") or [])]
        ts = _to_ms(book.get("timestamp", at_time or 0))
        return OrderBook(
            market_id=str(book.get("token_id", token_id)),
            timestamp_ms=ts,
            bids=bids,
            asks=asks,
        )

    # ------------------------------------------------------------------
    # Real-time WebSocket order stream
    # ------------------------------------------------------------------
    async def stream_trades(
        self,
        market_slugs: Optional[Iterable[str]] = None,
        callback: Optional[Callable[[Trade], Awaitable[None]]] = None,
        *,
        condition_ids: Optional[Iterable[str]] = None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        """Subscribe to the live order stream and await ``callback`` per trade.

        Subscribe by ``market_slugs`` or ``condition_ids`` (Dome does not support
        token-level WS filters). The API key is part of the connection URL.
        Reconnects with capped exponential backoff; pass ``stop_event`` to exit.
        """

        if callback is None:
            raise ValueError("callback is required")

        import websockets  # imported lazily so unit tests need no network deps

        url = f"{self.ws_url}/{self.api_key}" if self.api_key else self.ws_url
        market_slugs = list(market_slugs) if market_slugs else None
        condition_ids = list(condition_ids) if condition_ids else None

        filters: dict[str, Any]
        if condition_ids:
            filters = {"condition_ids": condition_ids}
        elif market_slugs:
            filters = {"market_slugs": market_slugs}
        else:
            filters = {"users": ["*"]}  # wildcard: all trades

        sub_msg = {
            "action": "subscribe",
            "platform": "polymarket",
            "version": 1,
            "type": "orders",
            "filters": filters,
        }

        attempt = 0
        while stop_event is None or not stop_event.is_set():
            try:
                async with websockets.connect(url) as ws:
                    attempt = 0
                    await ws.send(json.dumps(sub_msg))
                    logger.info("Subscribed to Dome order stream: {}", filters)
                    async for raw in ws:
                        if stop_event is not None and stop_event.is_set():
                            break
                        trade = self._parse_ws_message(raw)
                        if trade is not None:
                            await callback(trade)
            except Exception as exc:  # pragma: no cover - network path
                attempt += 1
                delay = min(self.backoff_base * (2**attempt), 30.0)
                logger.warning("WS stream error: {}; reconnecting in {:.1f}s", exc, delay)
                await asyncio.sleep(delay)

    def _parse_ws_message(self, raw: str | bytes) -> Trade | None:
        """Parse a Dome WS frame; return a Trade for ``event`` order frames only."""

        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(msg, dict):
            return None
        # Control frames: {"type":"ack",...}; data frames: {"type":"event","data":{...}}.
        if msg.get("type") != "event":
            return None
        data = msg.get("data")
        if not isinstance(data, dict) or "price" not in data:
            return None
        return self._parse_order(data)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "DomeClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
