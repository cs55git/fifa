"""Tests for the Polymarket native API client."""

from __future__ import annotations

import json

import httpx
import pytest

from data.models import OrderBook, Trade
from data.polymarket_client import PolymarketClient


def _make_client(handler) -> PolymarketClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport, timeout=10)
    return PolymarketClient(client=http_client, backoff_base=0.0)


# ---------------------------------------------------------------------------
# Gamma markets
# ---------------------------------------------------------------------------
def _market_row(slug="wc-bra-vs-arg", cond="0xcond"):
    return {
        "id": "1",
        "slug": slug,
        "conditionId": cond,
        "question": "Brazil vs Argentina",
        "closed": True,
        # Double-encoded JSON strings, index-matched.
        "outcomes": json.dumps(["Brazil", "Argentina"]),
        "clobTokenIds": json.dumps(["111", "222"]),
    }


def test_get_markets_parses_double_encoded_fields():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/markets")
        # tag_id should be passed through to the markets query.
        assert request.url.params.get("tag_id") == "82"
        if int(request.url.params.get("offset", 0)) > 0:
            return httpx.Response(200, json=[])
        return httpx.Response(200, json=[_market_row()])

    client = _make_client(handler)
    markets = client.get_markets(tag_id="82")
    assert len(markets) == 1
    m = markets[0]
    assert m["market_slug"] == "wc-bra-vs-arg"
    assert m["condition_id"] == "0xcond"
    assert m["tokens"] == [
        {"token_id": "111", "label": "Brazil"},
        {"token_id": "222", "label": "Argentina"},
    ]


def test_get_tag_id_resolves_and_caches():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if "/tags/slug/" in request.url.path:
            calls["n"] += 1
            assert request.url.path.endswith("/tags/slug/premier-league")
            return httpx.Response(200, json={"id": "82", "slug": "premier-league"})
        return httpx.Response(404, json={})

    client = _make_client(handler)
    assert client.get_tag_id("premier-league") == "82"
    assert client.get_tag_id("premier-league") == "82"  # cached
    assert calls["n"] == 1


def test_get_tag_id_unknown_returns_none():
    client = _make_client(lambda r: httpx.Response(404, json={"error": "not found"}))
    assert client.get_tag_id("does-not-exist") is None


def test_get_markets_resolves_tag_slug_to_id():
    def handler(request: httpx.Request) -> httpx.Response:
        if "/tags/slug/" in request.url.path:
            return httpx.Response(200, json={"id": "82"})
        assert request.url.params.get("tag_id") == "82"
        if int(request.url.params.get("offset", 0)) > 0:
            return httpx.Response(200, json=[])
        return httpx.Response(200, json=[_market_row()])

    client = _make_client(handler)
    markets = client.get_markets(tag_slug="premier-league")
    assert len(markets) == 1


def test_discover_league_markets_dedupes_and_annotates():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/tags/slug/premier-league"):
            return httpx.Response(200, json={"id": "82"})
        if path.endswith("/tags/slug/norway-eliteserien"):
            return httpx.Response(200, json={"id": "102651"})
        if path.endswith("/tags/slug/bogus-league"):
            return httpx.Response(404, json={})
        # /markets
        if int(request.url.params.get("offset", 0)) > 0:
            return httpx.Response(200, json=[])
        tag = request.url.params.get("tag_id")
        if tag == "82":
            # Shared market also appears under norway below -> dedupe to EPL.
            return httpx.Response(200, json=[_market_row(slug="epl-1", cond="0xshared")])
        if tag == "102651":
            return httpx.Response(
                200,
                json=[
                    _market_row(slug="epl-1", cond="0xshared"),  # duplicate
                    _market_row(slug="nor-1", cond="0xnor"),
                ],
            )
        return httpx.Response(200, json=[])

    client = _make_client(handler)
    markets = client.discover_league_markets(
        ["premier-league", "norway-eliteserien", "bogus-league"],
        tiers={"premier-league": "deep", "norway-eliteserien": "thin"},
        moneyline_only=False,  # validation tested separately below
    )
    by_cond = {m["condition_id"]: m for m in markets}
    assert set(by_cond) == {"0xshared", "0xnor"}
    # First league to yield the shared market wins (premier-league).
    assert by_cond["0xshared"]["league_slug"] == "premier-league"
    assert by_cond["0xshared"]["league_tier"] == "deep"
    assert by_cond["0xnor"]["league_slug"] == "norway-eliteserien"
    assert by_cond["0xnor"]["league_tier"] == "thin"


def _moneyline_row(event_slug, event_id, outcome, *, cond, git, sport="moneyline"):
    """A binary moneyline market belonging to a match event."""

    return {
        "id": cond,
        "slug": f"{event_slug}-{outcome}",
        "conditionId": cond,
        "question": f"{git} ({event_slug})?",
        "closed": False,
        "sportsMarketType": sport,
        "groupItemTitle": git,
        "outcomes": json.dumps(["Yes", "No"]),
        "clobTokenIds": json.dumps([f"{cond}a", f"{cond}b"]),
        "events": [{"id": event_id, "slug": event_slug, "title": event_slug}],
    }


def _three_way(event_slug, event_id, a, b):
    """Three markets (team A win, team B win, draw) for one match event."""

    return [
        _moneyline_row(event_slug, event_id, a, cond=f"0x{event_slug}-a", git=a.title()),
        _moneyline_row(event_slug, event_id, b, cond=f"0x{event_slug}-b", git=b.title()),
        _moneyline_row(event_slug, event_id, "draw", cond=f"0x{event_slug}-d", git="Draw"),
    ]


def test_filter_three_way_moneyline_keeps_only_complete_matches():
    from data.polymarket_client import PolymarketClient as _PC, filter_three_way_moneyline

    complete = _three_way("aus-tur", "1", "aus", "tur")
    incomplete = _three_way("bra-mar", "2", "bra", "mar")[:2]  # missing draw
    no_event = [_market_row(slug="lonely", cond="0xlonely")]  # no event grouping

    rows = complete + incomplete + no_event
    normalized = [_PC._normalize_market(r) for r in rows]
    kept = filter_three_way_moneyline(normalized)

    slugs = {m["market_slug"] for m in kept}
    assert slugs == {"aus-tur-aus", "aus-tur-tur", "aus-tur-draw"}


def test_discover_league_markets_validates_three_way():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/tags/slug/fifa-world-cup"):
            return httpx.Response(200, json={"id": "999"})
        assert request.url.params.get("sports_market_types") == "moneyline"
        if int(request.url.params.get("offset", 0)) > 0:
            return httpx.Response(200, json=[])
        # One complete 3-way event plus a stray 2-way event that must be dropped.
        rows = _three_way("aus-tur", "1", "aus", "tur")
        rows += _three_way("bra-mar", "2", "bra", "mar")[:2]
        return httpx.Response(200, json=rows)

    client = _make_client(handler)
    markets = client.discover_league_markets(["fifa-world-cup"], closed=False)
    event_slugs = {m["event_slug"] for m in markets}
    assert event_slugs == {"aus-tur"}
    assert len(markets) == 3


def test_get_markets_paginates_and_caps():
    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params.get("offset", 0))
        # Always return a full page so pagination would continue without the cap.
        rows = [
            {
                "id": str(offset + i),
                "slug": f"m{offset + i}",
                "conditionId": f"0x{offset + i}",
                "outcomes": json.dumps(["Yes", "No"]),
                "clobTokenIds": json.dumps([f"{offset + i}a", f"{offset + i}b"]),
            }
            for i in range(100)
        ]
        return httpx.Response(200, json=rows)

    client = _make_client(handler)
    markets = client.get_markets(max_markets=150)
    assert len(markets) == 150


# ---------------------------------------------------------------------------
# Data API trades
# ---------------------------------------------------------------------------
def _trade_row(asset="111", slug="wc-bra-vs-arg", ts=1_700_000_000, price=0.5, size=10):
    return {
        "proxyWallet": "0xabc",
        "side": "BUY",
        "asset": asset,
        "conditionId": "0xcond",
        "size": size,
        "price": price,
        "timestamp": ts,
        "outcome": "Brazil",
        "slug": slug,
    }


def test_get_trades_maps_and_paginates():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/trades")
        offset = int(request.url.params.get("offset", 0))
        limit = int(request.url.params.get("limit", 500))
        if offset == 0:
            return httpx.Response(200, json=[_trade_row(ts=1_700_000_000)] * limit)
        if offset == limit:
            return httpx.Response(200, json=[_trade_row(ts=1_700_000_060)])  # short page
        return httpx.Response(200, json=[])

    client = _make_client(handler)
    trades = client.get_trades("0xcond", limit=500)
    assert len(trades) == 501
    assert all(isinstance(t, Trade) for t in trades)
    assert trades[0].market_id == "111"  # asset -> market_id
    assert trades[0].slug == "wc-bra-vs-arg"
    assert trades[0].timestamp_ms == 1_700_000_000_000  # seconds -> ms
    assert trades[0].size == 10


def test_get_trades_stops_at_offset_cap_via_400():
    """A 400 'offset exceeded' must stop pagination, not discard collected trades."""

    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params.get("offset", 0))
        limit = int(request.url.params.get("limit", 500))
        if offset <= 3000:
            return httpx.Response(200, json=[_trade_row(ts=1_700_000_000)] * limit)
        return httpx.Response(
            400, json={"error": "max historical activity offset of 3000 exceeded"}
        )

    client = _make_client(handler)
    trades = client.get_trades("0xcond", limit=500)
    # Offsets 0..3000 (7 full pages of 500) succeed; the 3500 page 400s and stops.
    assert len(trades) == 3500
    assert all(isinstance(t, Trade) for t in trades)


def test_get_trades_stops_when_offset_exceeds_cap():
    """Even if the API kept serving rows, we stop once offset passes the cap."""

    def handler(request: httpx.Request) -> httpx.Response:
        limit = int(request.url.params.get("limit", 500))
        return httpx.Response(200, json=[_trade_row()] * limit)  # always a full page

    client = _make_client(handler)
    trades = client.get_trades("0xcond", limit=500)
    assert len(trades) == 3500  # capped at offset 3000 + final 500-row page


def test_iso_to_s_parses_polymarket_formats():
    from data.polymarket_client import _iso_to_s

    assert _iso_to_s("2026-04-06T22:28:02.973524Z") is not None
    assert _iso_to_s("2026-06-14 04:00:00+00") is not None
    assert _iso_to_s(None) is None
    assert _iso_to_s("not-a-date") is None
    # Both equivalent representations resolve to the same instant.
    assert _iso_to_s("2026-06-14T04:00:00Z") == _iso_to_s("2026-06-14 04:00:00+00")


def test_get_market_history_backfills_when_truncated():
    """When trades hit the offset cap, older history is pulled from prices-history."""

    boundary_ts = 1_700_000_000  # oldest real fill (seconds)

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/trades"):
            offset = int(request.url.params.get("offset", 0))
            limit = int(request.url.params.get("limit", 500))
            if offset <= 3000:  # always full pages -> triggers the cap stop
                return httpx.Response(
                    200, json=[_trade_row(asset="111", ts=boundary_ts)] * limit
                )
            return httpx.Response(400, json={"error": "offset exceeded"})
        if path.endswith("/prices-history"):
            # Two older points (kept) and one at/after the boundary (dropped).
            return httpx.Response(
                200,
                json={
                    "history": [
                        {"t": boundary_ts - 120, "p": 0.40},
                        {"t": boundary_ts - 60, "p": 0.35},
                        {"t": boundary_ts + 60, "p": 0.30},
                    ]
                },
            )
        return httpx.Response(404, json={})

    client = _make_client(handler)
    market = {
        "condition_id": "0xcond",
        "start_date": "2026-04-06T22:28:02Z",
        "end_date": "2026-04-11T18:00:00Z",
        "tokens": [{"token_id": "111", "label": "Yes"}],
    }
    trades = client.get_market_history(market, backfill=True)
    synthesized = [t for t in trades if t.size == 0.0]
    # Only the two points strictly older than the boundary are backfilled.
    assert len(synthesized) == 2
    assert all(t.timestamp_ms < boundary_ts * 1000 for t in synthesized)


def test_get_market_history_no_backfill_when_not_truncated():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/trades"):
            offset = int(request.url.params.get("offset", 0))
            if offset == 0:
                return httpx.Response(200, json=[_trade_row()] * 10)  # short page
            return httpx.Response(200, json=[])
        raise AssertionError("prices-history must not be called when not truncated")

    client = _make_client(handler)
    market = {"condition_id": "0xcond", "tokens": [{"token_id": "111"}]}
    trades = client.get_market_history(market, backfill=True)
    assert len(trades) == 10
    assert all(t.size != 0.0 for t in trades)


def test_get_trades_respects_max_trades():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[_trade_row()] * 500)

    client = _make_client(handler)
    trades = client.get_trades("0xcond", limit=500, max_trades=10)
    assert len(trades) == 10


def test_get_trades_handles_dict_envelope():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"trades": [_trade_row()]})

    client = _make_client(handler)
    trades = client.get_trades("0xcond")
    assert len(trades) == 1


# ---------------------------------------------------------------------------
# CLOB order book
# ---------------------------------------------------------------------------
def test_get_orderbook_parses_levels():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/book")
        return httpx.Response(
            200,
            json={
                "asset_id": "111",
                "timestamp": 1_700_000_000,
                "bids": [{"price": "0.50", "size": "100"}, {"price": "0.49", "size": "50"}],
                "asks": [{"price": "0.51", "size": "80"}],
            },
        )

    client = _make_client(handler)
    book = client.get_orderbook("111")
    assert isinstance(book, OrderBook)
    assert book.bids == [(0.5, 100.0), (0.49, 50.0)]
    assert book.asks == [(0.51, 80.0)]


def test_get_orderbook_404_returns_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    client = _make_client(handler)
    book = client.get_orderbook("111")
    assert book.bids == []
    assert book.asks == []


# ---------------------------------------------------------------------------
# Prices history fallback
# ---------------------------------------------------------------------------
def test_get_prices_history_synthesizes_trades():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/prices-history")
        return httpx.Response(
            200,
            json={"history": [{"t": 1_700_000_000, "p": 0.5}, {"t": 1_700_000_060, "p": 0.42}]},
        )

    client = _make_client(handler)
    pts = client.get_prices_history("111", start_ts=1_700_000_000, end_ts=1_700_000_100)
    assert len(pts) == 2
    assert pts[0].price == 0.5
    assert pts[1].timestamp_ms == 1_700_000_060_000
    assert pts[0].size == 0.0


# ---------------------------------------------------------------------------
# WebSocket parsing
# ---------------------------------------------------------------------------
def test_ws_parses_last_trade_price():
    client = _make_client(lambda r: httpx.Response(200, json={}))
    raw = json.dumps(
        [
            {
                "event_type": "last_trade_price",
                "asset_id": "111",
                "price": "0.33",
                "size": "5",
                "timestamp": "1700000000000",
            },
            {"event_type": "book", "asset_id": "111"},  # ignored
        ]
    )
    trades = client._parse_ws_message(raw, slug_map={"111": "wc-bra-vs-arg"})
    assert len(trades) == 1
    assert trades[0].price == 0.33
    assert trades[0].market_id == "111"
    assert trades[0].slug == "wc-bra-vs-arg"


def test_ws_ignores_non_trade_frames():
    client = _make_client(lambda r: httpx.Response(200, json={}))
    assert client._parse_ws_message('{"event_type": "book"}') == []
    assert client._parse_ws_message("not json") == []
