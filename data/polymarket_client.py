"""Client for Polymarket's native public APIs (no auth required for reads).

Three REST surfaces plus a CLOB WebSocket are used:

- Gamma  (``https://gamma-api.polymarket.com``) -- market/event discovery.
  ``GET /markets`` returns market metadata. ``outcomes`` and ``clobTokenIds`` are
  *double-encoded* JSON strings that must be parsed; ``clobTokenIds`` is
  index-matched to ``outcomes`` (one token id per outcome).
- Data   (``https://data-api.polymarket.com``) -- historical trade fills.
  ``GET /trades?market=<conditionId>`` returns executed fills with ``asset``
  (token id), ``price``, ``size`` and ``timestamp`` (Unix seconds).
- CLOB   (``https://clob.polymarket.com``) -- ``GET /book?token_id=`` for the live
  order book and ``GET /prices-history?market=<token_id>`` for historical prices.
  The CLOB WebSocket (``/ws/market``) streams ``last_trade_price`` events.

This is the default ingestion source for the bot. Internally we map
``Trade.market_id = token_id`` (the ERC-1155 outcome / "asset") and
``Trade.slug = market_slug`` so the rest of the pipeline is source-agnostic.
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

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# The Data API rejects historical trade queries past this offset with a 400
# ("max historical activity offset of 3000 exceeded"). Trades are returned
# newest-first, so with a 500-row page this still yields the most recent ~3500
# fills -- enough to cover a single match window for nearly all markets.
MAX_DATA_OFFSET = 3000


def _to_ms(ts: Any) -> int:
    try:
        ts = int(float(ts))
    except (TypeError, ValueError):
        return 0
    if 1_000_000_000 <= ts < 10_000_000_000:  # Unix seconds -> ms
        ts *= 1000
    return ts


def _iso_to_s(value: Any) -> Optional[int]:
    """Parse a Polymarket ISO timestamp to Unix seconds (tolerant of formats).

    Handles ``2026-04-06T22:28:02.973524Z`` and ``2026-06-14 04:00:00+00``.
    Returns None on failure.
    """

    if not value or not isinstance(value, str):
        return None
    import datetime as _dt

    s = value.strip().replace(" ", "T", 1)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    elif s.endswith("+00"):  # e.g. "...+00" -> "...+00:00"
        s = s + ":00"
    try:
        return int(_dt.datetime.fromisoformat(s).timestamp())
    except ValueError:
        return None


def _loads_maybe(value: Any) -> Any:
    """Parse Gamma's double-encoded JSON string fields; pass through real lists."""

    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _is_draw_market(market: dict) -> bool:
    """True if a normalized moneyline market is the draw outcome of its match."""

    git = (market.get("group_item_title") or "").strip().lower()
    if git == "draw" or git.startswith("draw "):  # "Draw", "Draw (A vs. B)"
        return True
    slug = (market.get("market_slug") or "").lower()
    if slug.endswith("-draw"):
        return True
    title = (market.get("title") or "").lower()
    return "in a draw" in title


def filter_three_way_moneyline(markets: list[dict]) -> list[dict]:
    """Keep only moneyline markets that belong to a valid 3-way soccer match.

    Markets are grouped by their event (``event_id``, falling back to
    ``event_slug``). A valid group has exactly three outcomes -- team A wins,
    team B wins and a draw -- so we require three markets in the group with
    exactly one draw. Markets without event grouping, or in incomplete/non-3-way
    groups (e.g. 2-way sports), are dropped. Input order is preserved.
    """

    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for m in markets:
        key = m.get("event_id") or m.get("event_slug")
        if not key:
            continue
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(m)

    valid: set[str] = set()
    for key, group in groups.items():
        draws = sum(1 for m in group if _is_draw_market(m))
        if len(group) == 3 and draws == 1:
            valid.add(key)

    return [
        m
        for m in markets
        if (m.get("event_id") or m.get("event_slug")) in valid
    ]


def _level(raw: Any) -> tuple[float, float]:
    if isinstance(raw, dict):
        return float(raw.get("price")), float(raw.get("size"))
    return float(raw[0]), float(raw[1])


class PolymarketClient:
    def __init__(
        self,
        *,
        gamma_url: Optional[str] = None,
        data_url: Optional[str] = None,
        clob_url: Optional[str] = None,
        clob_ws_url: Optional[str] = None,
        client: Optional[httpx.Client] = None,
        max_retries: int = 5,
        backoff_base: float = 0.5,
    ) -> None:
        self.gamma_url = (gamma_url or config.POLYMARKET_GAMMA_URL).rstrip("/")
        self.data_url = (data_url or config.POLYMARKET_DATA_URL).rstrip("/")
        self.clob_url = (clob_url or config.POLYMARKET_CLOB_URL).rstrip("/")
        self.clob_ws_url = clob_ws_url or config.POLYMARKET_CLOB_WS_URL
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._client = client or httpx.Client(
            headers={"Accept": "application/json"}, timeout=30.0
        )
        self._tag_id_cache: dict[str, Optional[str]] = {}

    # ------------------------------------------------------------------
    # Low-level GET with retry/backoff
    # ------------------------------------------------------------------
    def _get(
        self,
        url: str,
        params: Any = None,
        *,
        allow_404: bool = False,
        stop_400_on: Optional[str] = None,
    ) -> Any:
        attempt = 0
        while True:
            resp = self._client.get(url, params=params)
            if resp.status_code in _RETRYABLE_STATUS and attempt < self.max_retries:
                delay = self._retry_delay(resp, attempt)
                logger.warning(
                    "Polymarket GET {} -> {}; retry in {:.2f}s ({}/{})",
                    url,
                    resp.status_code,
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                self._sleep(delay)
                attempt += 1
                continue
            if allow_404 and resp.status_code == 404:
                return None
            # Some endpoints signal "no more data" with a 400 (e.g. the Data API's
            # offset cap). Treat such an expected 400 as a graceful stop.
            if (
                stop_400_on is not None
                and resp.status_code == 400
                and stop_400_on.lower() in resp.text.lower()
            ):
                return None
            resp.raise_for_status()
            if not resp.content:
                return {}
            return resp.json()

    def _retry_delay(self, resp: httpx.Response, attempt: int) -> float:
        ra = resp.headers.get("Retry-After")
        if ra:
            try:
                return float(ra)
            except ValueError:
                pass
        return self.backoff_base * (2**attempt)

    def _sleep(self, seconds: float) -> None:  # pragma: no cover - trivial
        import time

        time.sleep(seconds)

    # ------------------------------------------------------------------
    # Gamma: tag resolution
    # ------------------------------------------------------------------
    def get_tag_id(self, tag_slug: str) -> Optional[str]:
        """Resolve a tag slug (e.g. "premier-league") to its numeric tag id.

        The ``/markets`` endpoint filters by numeric ``tag_id``, not by slug, so
        league discovery must resolve slugs first. Results are cached; unknown
        slugs return ``None``.
        """

        if tag_slug in self._tag_id_cache:
            return self._tag_id_cache[tag_slug]
        payload = self._get(
            f"{self.gamma_url}/tags/slug/{tag_slug}", allow_404=True
        )
        tag_id = None
        if isinstance(payload, dict) and payload.get("id") is not None:
            tag_id = str(payload["id"])
        self._tag_id_cache[tag_slug] = tag_id
        return tag_id

    # ------------------------------------------------------------------
    # Gamma: market discovery
    # ------------------------------------------------------------------
    def get_markets(
        self,
        *,
        tag_id: Optional[str] = None,
        tag_slug: Optional[str] = None,
        closed: Optional[bool] = True,
        active: Optional[bool] = None,
        sports_market_types: Optional[str] = None,
        order: Optional[str] = "volume",
        ascending: bool = False,
        limit: int = 100,
        max_markets: Optional[int] = None,
    ) -> list[dict]:
        """Discover Polymarket markets, following offset pagination.

        Filter by numeric ``tag_id`` (the only tag filter ``/markets`` actually
        honors). A ``tag_slug`` is resolved to its id automatically.
        ``sports_market_types`` (e.g. ``"moneyline"``) narrows to a single sports
        market type. Returns normalized dicts with ``market_slug``,
        ``condition_id``, ``title``, ``closed``, event grouping fields and a
        ``tokens`` list of ``{"token_id", "label"}``.
        """

        if tag_id is None and tag_slug:
            tag_id = self.get_tag_id(tag_slug)

        out: list[dict] = []
        offset = 0
        page = min(limit, 500)
        while True:
            params: dict[str, Any] = {"limit": page, "offset": offset}
            if tag_id is not None:
                params["tag_id"] = tag_id
            if closed is not None:
                params["closed"] = str(closed).lower()
            if active is not None:
                params["active"] = str(active).lower()
            if sports_market_types is not None:
                params["sports_market_types"] = sports_market_types
            if order:
                params["order"] = order
                params["ascending"] = str(ascending).lower()

            payload = self._get(f"{self.gamma_url}/markets", params=params)
            rows = payload if isinstance(payload, list) else payload.get("data", [])
            if not rows:
                break
            for m in rows:
                out.append(self._normalize_market(m))
                if max_markets is not None and len(out) >= max_markets:
                    return out
            if len(rows) < page:
                break
            offset += page
        return out

    def discover_league_markets(
        self,
        leagues: Iterable[str],
        *,
        tiers: Optional[dict[str, str]] = None,
        closed: Optional[bool] = True,
        max_markets_per_league: Optional[int] = None,
        moneyline_only: bool = True,
    ) -> list[dict]:
        """Discover markets across multiple soccer leagues, deduped by market.

        For each league slug, resolves its tag id and queries ``/markets`` by that
        id, then annotates each market with ``league_slug`` and ``league_tier``
        (from ``tiers``). When ``moneyline_only`` is set we request only moneyline
        markets and keep only those belonging to a valid 3-way match (team A win,
        team B win, draw); other market types (spreads, totals, player props) and
        incomplete groups are dropped. Markets are deduped by ``condition_id``
        (first league that yields a market wins). Unknown league slugs are skipped
        with a warning.
        """

        tiers = tiers or {}
        seen: set[str] = set()
        out: list[dict] = []
        for league in leagues:
            tag_id = self.get_tag_id(league)
            if tag_id is None:
                logger.warning("Unknown league tag slug '{}', skipping", league)
                continue
            # Fetch the full set first so 3-way validation sees complete events;
            # cap per league only after validation.
            markets = self.get_markets(
                tag_id=tag_id,
                closed=closed,
                sports_market_types="moneyline" if moneyline_only else None,
            )
            raw_count = len(markets)
            if moneyline_only:
                markets = filter_three_way_moneyline(markets)

            kept = 0
            for m in markets:
                cond = m.get("condition_id") or m.get("market_slug")
                if cond in seen:
                    continue
                if max_markets_per_league is not None and kept >= max_markets_per_league:
                    break
                seen.add(cond)
                m["league_slug"] = league
                m["league_tier"] = tiers.get(league)
                out.append(m)
                kept += 1
            logger.info(
                "League '{}' (tag {}): {} moneyline markets -> {} valid 3-way -> {} kept",
                league,
                tag_id,
                raw_count,
                len(markets),
                kept,
            )
        return out

    @staticmethod
    def _normalize_market(m: dict) -> dict:
        outcomes = _loads_maybe(m.get("outcomes", "[]")) or []
        token_ids = _loads_maybe(m.get("clobTokenIds", "[]")) or []
        tokens = []
        for i, tid in enumerate(token_ids):
            label = outcomes[i] if i < len(outcomes) else ""
            tokens.append({"token_id": str(tid), "label": label})
        events = m.get("events") or []
        event = events[0] if isinstance(events, list) and events else {}
        return {
            "market_slug": m.get("slug", ""),
            "condition_id": m.get("conditionId", ""),
            "title": m.get("question", m.get("title", "")),
            "closed": m.get("closed"),
            "volume": m.get("volume"),
            "start_date": m.get("startDate"),
            "end_date": m.get("endDate"),
            "game_start_time": m.get("gameStartTime"),
            "sports_market_type": m.get("sportsMarketType"),
            "group_item_title": m.get("groupItemTitle"),
            "event_id": str(event.get("id")) if event.get("id") is not None else None,
            "event_slug": event.get("slug"),
            "event_title": event.get("title"),
            "tokens": tokens,
            "raw": m,
        }

    # ------------------------------------------------------------------
    # Data API: historical trade fills
    # ------------------------------------------------------------------
    def get_trades(
        self,
        condition_id: str,
        *,
        limit: int = 500,
        max_trades: Optional[int] = None,
        taker_only: bool = False,
        meta: Optional[dict] = None,
    ) -> list[Trade]:
        """Fetch executed trade fills for a market (by condition id), paginated.

        Each fill becomes a :class:`Trade` keyed by its outcome token
        (``market_id = asset``). The Data API returns trades newest-first and
        paginates by ``offset`` but rejects offsets past :data:`MAX_DATA_OFFSET`
        with a 400. We stop cleanly at that cap (keeping the most recent ~3500
        fills) rather than letting the error discard the whole market; we also
        stop on a short page or when ``max_trades`` is reached.
        """

        trades: list[Trade] = []
        offset = 0
        page = min(limit, 10_000)
        truncated = False
        while True:
            params = {
                "market": condition_id,
                "limit": page,
                "offset": offset,
                "takerOnly": str(taker_only).lower(),
            }
            payload = self._get(
                f"{self.data_url}/trades", params=params, stop_400_on="offset"
            )
            if payload is None:  # offset cap hit defensively (400)
                truncated = True
                break
            rows = payload if isinstance(payload, list) else payload.get("trades", [])
            if not rows:
                break
            for row in rows:
                trades.append(self._parse_trade(row))
                if max_trades is not None and len(trades) >= max_trades:
                    return trades
            if len(rows) < page:
                break
            offset += page
            if offset > MAX_DATA_OFFSET:
                truncated = True
                break

        if truncated:
            logger.info(
                "Data API offset cap reached for {}: returning {} most-recent "
                "trades (older fills beyond the {}-offset limit are unavailable)",
                condition_id,
                len(trades),
                MAX_DATA_OFFSET,
            )
        if meta is not None:
            meta["truncated"] = truncated
        return trades

    def get_market_history(
        self,
        market: dict,
        *,
        max_trades: Optional[int] = None,
        backfill: bool = False,
    ) -> list[Trade]:
        """Fetch a market's full trade history, optionally backfilling the cap.

        Real fills are fetched first (newest ~3500). When the Data API offset cap
        truncates them and ``backfill`` is set, the older portion of each outcome
        token is reconstructed from the CLOB ``prices-history`` endpoint (1-minute
        fidelity over the market's date range), synthesizing price-point trades
        for the window *before* the oldest real fill. Synthesized trades carry
        ``size=0``; resolution is coarser than real fills, so shock depths in the
        backfilled region may be slightly understated.
        """

        condition_id = market.get("condition_id")
        if not condition_id:
            return []
        meta: dict = {}
        trades = self.get_trades(condition_id, max_trades=max_trades, meta=meta)
        if not (backfill and meta.get("truncated")):
            return trades

        # Oldest real fill per outcome token -> the boundary to backfill up to.
        oldest_by_token: dict[str, int] = {}
        for t in trades:
            cur = oldest_by_token.get(t.market_id)
            if cur is None or t.timestamp_ms < cur:
                oldest_by_token[t.market_id] = t.timestamp_ms

        start_s = _iso_to_s(market.get("game_start_time") or market.get("start_date"))
        end_s = _iso_to_s(market.get("end_date"))
        for token in market.get("tokens", []):
            tid = str(token.get("token_id"))
            boundary_ms = oldest_by_token.get(tid)
            query_end = (boundary_ms // 1000) if boundary_ms else end_s
            if start_s is None or query_end is None:
                continue
            try:
                points = self.get_prices_history(
                    tid, start_ts=start_s, end_ts=query_end
                )
            except Exception as exc:  # backfill is best-effort
                logger.warning("Backfill prices-history failed for {}: {}", tid, exc)
                continue
            added = 0
            for p in points:
                if boundary_ms is None or p.timestamp_ms < boundary_ms:
                    trades.append(p)
                    added += 1
            if added:
                logger.info("Backfilled {} older price points for token {}", added, tid)
        return trades

    @staticmethod
    def _parse_trade(row: dict) -> Trade:
        return Trade(
            market_id=str(row.get("asset", "")),
            slug=str(row.get("slug", "")),
            timestamp_ms=_to_ms(row.get("timestamp", 0)),
            price=float(row.get("price", 0.0)),
            size=float(row.get("size", 0.0)),
        )

    # ------------------------------------------------------------------
    # CLOB: order book + prices history
    # ------------------------------------------------------------------
    def get_orderbook(self, token_id: str) -> OrderBook:
        """Fetch the live order book for a token. Resolved markets 404 -> empty."""

        payload = self._get(
            f"{self.clob_url}/book", params={"token_id": token_id}, allow_404=True
        )
        if not isinstance(payload, dict) or "bids" not in payload:
            return OrderBook(market_id=token_id, timestamp_ms=0, bids=[], asks=[])
        bids = [_level(b) for b in (payload.get("bids") or [])]
        asks = [_level(a) for a in (payload.get("asks") or [])]
        return OrderBook(
            market_id=str(payload.get("asset_id", token_id)),
            timestamp_ms=_to_ms(payload.get("timestamp", 0)),
            bids=bids,
            asks=asks,
        )

    def get_prices_history(
        self,
        token_id: str,
        *,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        interval: Optional[str] = None,
        fidelity: int = 1,
    ) -> list[Trade]:
        """Fetch historical prices and synthesize price-point "trades".

        Useful as a fallback for markets without granular trade data. ``interval``
        and ``start_ts``/``end_ts`` are mutually exclusive; for resolved markets
        prefer an explicit timestamp range. Synthesized trades carry ``size=0``.
        """

        params: dict[str, Any] = {"market": token_id, "fidelity": fidelity}
        if interval is not None:
            params["interval"] = interval
        else:
            if start_ts is not None:
                params["startTs"] = start_ts
            if end_ts is not None:
                params["endTs"] = end_ts
        payload = self._get(f"{self.clob_url}/prices-history", params=params)
        history = payload.get("history", []) if isinstance(payload, dict) else []
        return [
            Trade(
                market_id=str(token_id),
                slug="",
                timestamp_ms=_to_ms(pt.get("t", 0)),
                price=float(pt.get("p", 0.0)),
                size=0.0,
            )
            for pt in history
        ]

    # ------------------------------------------------------------------
    # CLOB WebSocket: live trade stream (last_trade_price events)
    # ------------------------------------------------------------------
    async def stream_trades(
        self,
        token_ids: Iterable[str],
        callback: Callable[[Trade], Awaitable[None]],
        *,
        slug_map: Optional[dict[str, str]] = None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        """Subscribe to the CLOB market channel for ``token_ids`` (asset ids).

        Emits a :class:`Trade` for every ``last_trade_price`` event. ``slug_map``
        optionally maps token_id -> market_slug so emitted trades carry a slug for
        league classification. Reconnects with capped backoff.
        """

        import websockets  # lazy import so unit tests need no network deps

        token_ids = list(token_ids)
        slug_map = slug_map or {}
        sub_msg = {"type": "market", "assets_ids": token_ids}

        attempt = 0
        while stop_event is None or not stop_event.is_set():
            try:
                async with websockets.connect(self.clob_ws_url) as ws:
                    attempt = 0
                    await ws.send(json.dumps(sub_msg))
                    logger.info("Subscribed to CLOB market channel for {} tokens", len(token_ids))
                    async for raw in ws:
                        if stop_event is not None and stop_event.is_set():
                            break
                        for trade in self._parse_ws_message(raw, slug_map):
                            await callback(trade)
            except Exception as exc:  # pragma: no cover - network path
                attempt += 1
                delay = min(self.backoff_base * (2**attempt), 30.0)
                logger.warning("CLOB WS error: {}; reconnecting in {:.1f}s", exc, delay)
                await asyncio.sleep(delay)

    def _parse_ws_message(
        self, raw: str | bytes, slug_map: Optional[dict[str, str]] = None
    ) -> list[Trade]:
        """Parse a CLOB WS frame into zero or more trades (last_trade_price only)."""

        slug_map = slug_map or {}
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
        events = msg if isinstance(msg, list) else [msg]
        trades: list[Trade] = []
        for ev in events:
            if not isinstance(ev, dict):
                continue
            if ev.get("event_type") != "last_trade_price":
                continue
            asset = str(ev.get("asset_id", ev.get("asset", "")))
            if "price" not in ev:
                continue
            trades.append(
                Trade(
                    market_id=asset,
                    slug=slug_map.get(asset, ""),
                    timestamp_ms=_to_ms(ev.get("timestamp", 0)),
                    price=float(ev.get("price", 0.0)),
                    size=float(ev.get("size", 0.0) or 0.0),
                )
            )
        return trades

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "PolymarketClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
