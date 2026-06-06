"""Tests for the data ingestion layer (Dome client + DuckDB storage)."""

from __future__ import annotations

import json

import httpx
import pytest

from data.dome_client import DomeClient
from data.models import OrderBook, ShockEvent, Trade
from data.storage import Storage


def _make_client(handler) -> DomeClient:
    """Build a DomeClient backed by an httpx MockTransport."""

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(
        base_url="https://api.dome.test/v1",
        transport=transport,
    )
    return DomeClient(
        api_key="test-key",
        base_url="https://api.dome.test/v1",
        client=http_client,
        backoff_base=0.0,
    )


def _order(token_id="tok1", slug="epl-game", ts=1_700_000_000, price=0.5, shares=100.0):
    return {
        "token_id": token_id,
        "token_label": "Yes",
        "side": "BUY",
        "market_slug": slug,
        "condition_id": "0xcond",
        "shares": shares * 1_000_000,
        "shares_normalized": shares,
        "price": price,
        "timestamp": ts,  # Unix seconds
    }


def test_get_trades_returns_trade_objects():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/polymarket/orders")
        assert request.headers.get("x-api-key") == "test-key"
        return httpx.Response(
            200,
            json={
                "orders": [
                    _order(ts=1_700_000_000, price=0.5, shares=100),
                    _order(ts=1_700_000_060, price=0.45, shares=50),
                ],
                "pagination": {"has_more": False},
            },
        )

    client = _make_client(handler)
    trades = client.get_trades(token_id="tok1")
    assert len(trades) == 2
    assert all(isinstance(t, Trade) for t in trades)
    assert trades[0].price == 0.5
    assert trades[0].market_id == "tok1"
    assert trades[0].slug == "epl-game"
    assert trades[1].size == 50  # shares_normalized
    assert trades[0].timestamp_ms == 1_700_000_000_000  # seconds -> ms


def test_get_trades_follows_pagination():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        key = request.url.params.get("pagination_key")
        if key is None:
            return httpx.Response(
                200,
                json={
                    "orders": [_order(ts=1_700_000_000, price=0.5)],
                    "pagination": {"has_more": True, "pagination_key": "page2"},
                },
            )
        return httpx.Response(
            200,
            json={
                "orders": [_order(ts=1_700_000_060, price=0.4)],
                "pagination": {"has_more": False},
            },
        )

    client = _make_client(handler)
    trades = client.get_trades(token_id="tok1")
    assert calls["n"] == 2
    assert len(trades) == 2
    assert [t.timestamp_ms for t in trades] == [1_700_000_000_000, 1_700_000_060_000]


def test_get_trades_requires_exactly_one_filter():
    client = _make_client(lambda r: httpx.Response(200, json={"orders": []}))
    import pytest as _pytest

    with _pytest.raises(ValueError):
        client.get_trades()  # no filter
    with _pytest.raises(ValueError):
        client.get_trades(token_id="t", market_slug="s")  # two filters


def test_rate_limit_retry_with_backoff():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={})
        return httpx.Response(
            200,
            json={"orders": [_order()], "pagination": {"has_more": False}},
        )

    client = _make_client(handler)
    trades = client.get_trades(token_id="tok1")
    assert calls["n"] == 2  # one 429 then one success
    assert len(trades) == 1


def test_get_markets_normalizes_sides():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/polymarket/markets")
        return httpx.Response(
            200,
            json={
                "markets": [
                    {
                        "market_slug": "wc-bra-vs-arg",
                        "condition_id": "0xabc",
                        "title": "Brazil vs Argentina",
                        "tags": ["Soccer", "World Cup"],
                        "status": "open",
                        "side_a": {"id": "111", "label": "Brazil"},
                        "side_b": {"id": "222", "label": "Argentina"},
                    }
                ],
                "pagination": {"has_more": False},
            },
        )

    client = _make_client(handler)
    markets = client.get_markets(tags=["Soccer"])
    assert len(markets) == 1
    m = markets[0]
    assert m["market_slug"] == "wc-bra-vs-arg"
    assert {t["token_id"] for t in m["tokens"]} == {"111", "222"}


def test_get_orderbook_parses_levels():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/polymarket/orderbook")
        return httpx.Response(
            200,
            json={
                "token_id": "tok1",
                "timestamp": 1_700_000_000,
                "bids": [[0.5, 100], {"price": 0.49, "size": 50}],
                "asks": [[0.51, 80]],
            },
        )

    client = _make_client(handler)
    book = client.get_orderbook("tok1")
    assert isinstance(book, OrderBook)
    assert book.bids == [(0.5, 100.0), (0.49, 50.0)]
    assert book.asks == [(0.51, 80.0)]
    assert book.timestamp_ms == 1_700_000_000_000


def test_ws_message_parsing():
    client = _make_client(lambda r: httpx.Response(200, json={}))
    trade = client._parse_ws_message(
        json.dumps(
            {
                "type": "event",
                "subscription_id": "sub_1",
                "data": _order(token_id="tok9", price=0.3, shares=5, ts=1_700_000_000),
            }
        )
    )
    assert trade is not None
    assert trade.price == 0.3
    assert trade.market_id == "tok9"
    assert trade.size == 5
    # Acknowledgment / control frames are ignored.
    assert client._parse_ws_message('{"type": "ack", "subscription_id": "sub_1"}') is None


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def test_init_db_creates_all_tables():
    store = Storage(db_path=":memory:")
    tables = store.table_names()
    assert {"trades", "orderbooks", "shocks", "orders"} <= tables
    store.close()


def test_insert_and_get_trades_roundtrip():
    store = Storage(db_path=":memory:")
    trades = [
        Trade(market_id="m1", slug="s", timestamp_ms=3000, price=0.4, size=1),
        Trade(market_id="m1", slug="s", timestamp_ms=1000, price=0.5, size=1),
        Trade(market_id="m2", slug="s", timestamp_ms=2000, price=0.6, size=1),
    ]
    store.insert_trades(trades)
    out = store.get_trades("m1")
    assert len(out) == 2
    # Ordered ascending by timestamp.
    assert [t.timestamp_ms for t in out] == [1000, 3000]
    assert store.list_market_ids() == ["m1", "m2"]
    store.close()


def test_get_trades_time_window():
    store = Storage(db_path=":memory:")
    store.insert_trades(
        [
            Trade(market_id="m1", slug="s", timestamp_ms=t, price=0.5, size=1)
            for t in (1000, 2000, 3000, 4000)
        ]
    )
    out = store.get_trades("m1", start_ms=2000, end_ms=3000)
    assert [t.timestamp_ms for t in out] == [2000, 3000]
    store.close()


def test_insert_shock_and_order():
    store = Storage(db_path=":memory:")
    shock = ShockEvent(
        market_id="m1",
        slug="epl-game",
        peak=0.5,
        floor=0.35,
        depth=0.15,
        pre_price=0.5,
        elapsed_ms=35 * 60_000,
        goal_diff=0,
        bids=[(0.5, 100)],
        bucket_key="deep|balanced|balanced|mid|level",
        detected_at_ms=123,
    )
    store.insert_shock(shock)
    store.insert_order(
        {
            "order_id": "o1",
            "market_id": "m1",
            "bucket_key": shock.bucket_key,
            "percentile": 50,
            "price": 0.22,
            "size": 5.0,
            "status": "open",
            "created_at_ms": 123,
            "updated_at_ms": 123,
        }
    )
    n_shocks = store.con.execute("SELECT COUNT(*) FROM shocks").fetchone()[0]
    n_orders = store.con.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    assert n_shocks == 1
    assert n_orders == 1
    store.close()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
