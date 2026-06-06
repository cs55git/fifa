"""DuckDB-backed persistence layer.

Stores raw trades, order-book snapshots, detected shocks and order records.
DuckDB is used because it is an embedded, file-based analytical database that
needs no server and is fast for the columnar replay queries the distribution
builder and backtester perform.
"""

from __future__ import annotations

import json
from typing import Any

import duckdb

import config
from data.models import OrderBook, ShockEvent, Trade


class Storage:
    """Wrapper around a DuckDB connection with the bot's schema."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or config.DB_PATH
        self.con = duckdb.connect(self.db_path)
        self.init_db()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def init_db(self) -> None:
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                market_id    VARCHAR,
                slug         VARCHAR,
                timestamp_ms BIGINT,
                price        DOUBLE,
                size         DOUBLE
            )
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS orderbooks (
                market_id    VARCHAR,
                timestamp_ms BIGINT,
                bids         VARCHAR,
                asks         VARCHAR
            )
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS shocks (
                market_id     VARCHAR,
                slug          VARCHAR,
                peak          DOUBLE,
                floor         DOUBLE,
                depth         DOUBLE,
                pre_price     DOUBLE,
                elapsed_ms    BIGINT,
                goal_diff     INTEGER,
                bucket_key    VARCHAR,
                detected_at_ms BIGINT
            )
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id      VARCHAR,
                market_id     VARCHAR,
                bucket_key    VARCHAR,
                percentile    INTEGER,
                price         DOUBLE,
                size          DOUBLE,
                status        VARCHAR,
                fill_price    DOUBLE,
                exit_price    DOUBLE,
                pnl           DOUBLE,
                created_at_ms BIGINT,
                updated_at_ms BIGINT
            )
            """
        )

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------
    def insert_trades(self, trades: list[Trade]) -> None:
        if not trades:
            return
        rows = [
            (t.market_id, t.slug, t.timestamp_ms, t.price, t.size) for t in trades
        ]
        self.con.executemany(
            "INSERT INTO trades VALUES (?, ?, ?, ?, ?)",
            rows,
        )

    def get_trades(
        self,
        market_id: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[Trade]:
        query = "SELECT market_id, slug, timestamp_ms, price, size FROM trades WHERE market_id = ?"
        params: list[Any] = [market_id]
        if start_ms is not None:
            query += " AND timestamp_ms >= ?"
            params.append(start_ms)
        if end_ms is not None:
            query += " AND timestamp_ms <= ?"
            params.append(end_ms)
        query += " ORDER BY timestamp_ms ASC"
        rows = self.con.execute(query, params).fetchall()
        return [
            Trade(
                market_id=r[0],
                slug=r[1],
                timestamp_ms=r[2],
                price=r[3],
                size=r[4],
            )
            for r in rows
        ]

    def list_market_ids(self) -> list[str]:
        rows = self.con.execute(
            "SELECT DISTINCT market_id FROM trades ORDER BY market_id"
        ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Order books
    # ------------------------------------------------------------------
    def insert_orderbook(self, book: OrderBook) -> None:
        self.con.execute(
            "INSERT INTO orderbooks VALUES (?, ?, ?, ?)",
            [
                book.market_id,
                book.timestamp_ms,
                json.dumps(book.bids),
                json.dumps(book.asks),
            ],
        )

    # ------------------------------------------------------------------
    # Shocks
    # ------------------------------------------------------------------
    def insert_shock(self, shock: ShockEvent) -> None:
        self.con.execute(
            "INSERT INTO shocks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                shock.market_id,
                shock.slug,
                shock.peak,
                shock.floor,
                shock.depth,
                shock.pre_price,
                shock.elapsed_ms,
                shock.goal_diff,
                shock.bucket_key,
                shock.detected_at_ms,
            ],
        )

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    def insert_order(self, order: dict[str, Any]) -> None:
        self.con.execute(
            """
            INSERT INTO orders (
                order_id, market_id, bucket_key, percentile, price, size,
                status, fill_price, exit_price, pnl, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                order.get("order_id"),
                order.get("market_id"),
                order.get("bucket_key"),
                order.get("percentile"),
                order.get("price"),
                order.get("size"),
                order.get("status", "open"),
                order.get("fill_price"),
                order.get("exit_price"),
                order.get("pnl"),
                order.get("created_at_ms"),
                order.get("updated_at_ms"),
            ],
        )

    def table_names(self) -> set[str]:
        rows = self.con.execute("SHOW TABLES").fetchall()
        return {r[0] for r in rows}

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
