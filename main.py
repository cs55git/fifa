"""Entry point for the FIFA World Cup shock trading bot.

Modes
-----
ingest    Pull historical football trades from the data source into the local DB.
build     Rebuild distributions.json from historical trades in the local DB.
backtest  Replay historical trades through the full pipeline and print a report.
paper     Run the live detector, executing into the paper engine (no real money).
live      Run the live detector, executing real orders on Polymarket (DANGER).

The data source defaults to Polymarket's native APIs (``--source polymarket``);
``--source dome`` is kept as a fallback. Live mode is gated behind
``PAPER_MODE=false`` in the environment to prevent accidental real-capital trading.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from loguru import logger

import config
from data.storage import Storage
from distributions import store
from distributions.builder import DistributionBuilder


# ---------------------------------------------------------------------------
# Data-source helpers
# ---------------------------------------------------------------------------
def _make_client(source: str):
    """Construct the configured data client (polymarket | dome)."""

    if source == "dome":
        from data.dome_client import DomeClient

        return DomeClient()
    from data.polymarket_client import PolymarketClient

    return PolymarketClient()


def _discover_markets(client, args, *, closed_override: bool | None = None) -> list[dict]:
    """Return normalized market dicts ({market_slug, condition_id, tokens, ...}).

    For Polymarket we restrict to ``sports_market_types=moneyline`` and validate
    each match has the full 3-way outcome set (team A win, team B win, draw),
    which keeps only true match-outcome markets and drops spreads, totals and
    player props. Pass ``--all-market-types`` to disable this.
    """

    moneyline_only = not getattr(args, "all_market_types", False)

    if args.source == "dome":
        if args.search:
            return client.get_markets(search=args.search, status=args.status)
        return client.get_markets(tags=args.tags, status=args.status)

    # Polymarket (Gamma).
    if closed_override is not None:
        closed = closed_override
    else:
        closed = None if args.status is None else (args.status.lower() == "closed")

    # Preferred path: discover per soccer league via numeric tag ids (the generic
    # "soccer" tag is too noisy / not honored by /markets).
    if args.leagues:
        return client.discover_league_markets(
            args.leagues,
            tiers=config.SOCCER_LEAGUE_TIERS,
            closed=closed,
            max_markets_per_league=args.max_markets,
            moneyline_only=moneyline_only,
        )

    # Fallback: a single explicit tag slug.
    from data.polymarket_client import filter_three_way_moneyline

    markets = client.get_markets(
        tag_slug=args.tag_slug,
        closed=closed,
        sports_market_types="moneyline" if moneyline_only else None,
        max_markets=None if moneyline_only else args.max_markets,
    )
    if moneyline_only:
        markets = filter_three_way_moneyline(markets)
        if args.max_markets is not None:
            markets = markets[: args.max_markets]
    return markets


def _annotate_slug(market: dict) -> str:
    """Prefix a market slug with its league slug so classify_league can read it.

    Soccer market slugs omit the league, so we fold the discovered league slug in
    (e.g. "premier-league/nor-sar-bog-..."). No-op when the league is unknown.
    """

    base = market.get("market_slug", "")
    league = market.get("league_slug")
    return f"{league}/{base}" if league else base


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def cmd_ingest(args: argparse.Namespace) -> int:
    """Pull historical football trades from the data source into the local DB.

    Discovers markets, then pulls trade history and stores it. Polymarket's Data
    API returns all fills for a market in one query (keyed per outcome token), so
    we fetch by ``condition_id``; Dome is fetched per outcome ``token_id``.
    """

    storage = Storage(args.db)
    client = _make_client(args.source)
    markets = _discover_markets(client, args)
    logger.info("[{}] discovered {} markets", args.source, len(markets))

    total_trades = 0
    for market in markets:
        slug = market["market_slug"]
        annotated = _annotate_slug(market)
        condition_id = market.get("condition_id")
        try:
            if args.source == "dome":
                trades = []
                for token in market["tokens"]:
                    trades += client.get_trades(
                        token_id=token["token_id"], start_ts=args.start, end_ts=args.end
                    )
            else:
                if not condition_id:
                    continue
                trades = client.get_market_history(
                    market, max_trades=args.max_trades, backfill=args.backfill
                )
        except Exception as exc:  # keep going on per-market failures
            logger.warning("Failed to fetch {} ({}): {}", slug, condition_id, exc)
            continue

        # Fold the league into the stored slug so downstream classification can
        # read the league tier (the raw market slug omits it).
        for t in trades:
            t.slug = annotated
        storage.insert_trades(trades)
        total_trades += len(trades)
        logger.info("{}: {} trades", annotated, len(trades))

    logger.info("Ingest complete: {} trades stored to {}", total_trades, args.db)
    client.close()
    storage.close()
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """Build per-bucket percentile distributions from stored historical trades."""

    storage = Storage(args.db)
    builder = DistributionBuilder()
    market_ids = storage.list_market_ids()
    logger.info("Building distributions from {} markets", len(market_ids))
    total = 0
    for market_id in market_ids:
        trades = storage.get_trades(market_id)
        total += builder.process_market_trades(trades)
    dists = builder.build()
    store.save(dists, args.out)
    logger.info(
        "Recorded {} shocks across {} buckets -> {}", total, len(dists), args.out
    )
    storage.close()
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    from backtest.runner import BacktestRunner

    storage = Storage(args.db)
    dists = store.load(args.dist) if _exists(args.dist) else {}
    runner = BacktestRunner(storage, dists, bucket_filter=args.buckets)
    runner.run()
    runner.report()
    storage.close()
    return 0


def cmd_gridsearch(args: argparse.Namespace) -> int:
    """Sweep take-profit / stop-loss combinations over the stored backtest data.

    Detection/classification are independent of the exit parameters, but the
    fill/exit simulation is not, so each (tp, sl) pair is replayed end-to-end.
    Prints a table sorted by total PnL.
    """

    import pandas as pd

    from backtest.runner import BacktestRunner

    storage = Storage(args.db)
    dists = store.load(args.dist) if _exists(args.dist) else {}

    # Preload trades once so every combo replays the same in-memory data.
    # --max-markets samples the first N markets to keep the sweep fast.
    market_ids = storage.list_market_ids()
    if args.max_markets is not None:
        market_ids = market_ids[: args.max_markets]
    all_trades = []
    for mid in market_ids:
        all_trades.extend(storage.get_trades(mid))
    logger.info(
        "Grid search over {} TP x {} SL x {} hold on {} trades from {} markets",
        len(args.tp_grid),
        len(args.sl_grid),
        len(args.hold_grid),
        len(all_trades),
        len(market_ids),
    )

    rows = []
    for tp in args.tp_grid:
        for sl in args.sl_grid:
            for hold in args.hold_grid:
                stop = None if sl <= 0 else sl
                max_hold = None if hold <= 0 else hold
                runner = BacktestRunner(
                    storage,
                    dists,
                    exit_cents=tp,
                    stop_loss_cents=stop,
                    max_hold_seconds=max_hold,
                    bucket_filter=args.buckets,
                )
                runner.run_trades(all_trades)
                s = runner.engine.summary()
                rows.append(
                    {
                        "take_profit": tp,
                        "stop_loss": "off" if stop is None else stop,
                        "max_hold_s": "off" if max_hold is None else max_hold,
                        "shocks": runner.n_shocks,
                        "orders": s["total_orders"],
                        "fill_rate": round(s["fill_rate"], 3),
                        "win_rate": round(s["win_rate"], 3),
                        "total_pnl": round(s["total_pnl"], 2),
                    }
                )

    table = pd.DataFrame(rows).sort_values("total_pnl", ascending=False).reset_index(drop=True)
    logger.info("Grid search results (best PnL first):\n{}", table.to_string(index=False))
    storage.close()
    return 0


def cmd_paper(args: argparse.Namespace) -> int:
    return _run_live(args, paper=True)


def cmd_live(args: argparse.Namespace) -> int:
    if config.PAPER_MODE:
        logger.error(
            "Refusing to run live: PAPER_MODE is true. Set PAPER_MODE=false to "
            "trade real capital."
        )
        return 1
    return _run_live(args, paper=False)


def _run_live(args: argparse.Namespace, *, paper: bool) -> int:
    from detector.live import LiveDetector, MarketContext
    from execution.ladder import build_ladder
    from execution.paper import PaperEngine

    storage = Storage(args.db)
    dists = store.load(args.dist) if _exists(args.dist) else {}
    client = _make_client(args.source)

    engine = PaperEngine(storage)
    if not paper:
        from execution.live_exec import LiveExecution

        engine = LiveExecution(storage)

    from classifier.bucket import bucket_matches

    buckets = getattr(args, "buckets", None)

    async def on_shock(event) -> None:
        if not bucket_matches(event.bucket_key, buckets):
            logger.debug("Skipping shock in filtered-out bucket {}", event.bucket_key)
            return
        percentiles = store.lookup(event.bucket_key, dists)
        orders = build_ladder(event.pre_price, percentiles, config.CAPITAL_PER_SHOCK)
        engine.submit(event, orders)

    # Discover live (open) markets and what to subscribe to.
    status = "open"
    if args.source == "dome":
        markets = client.get_markets(tags=args.tags, status=status)
        subscriptions = [m["market_slug"] for m in markets if m["market_slug"]]
        detector = LiveDetector(dists, on_shock, dome_client=client)
    else:
        markets = _discover_markets(client, args, closed_override=False)
        slug_map: dict[str, str] = {}
        subscriptions = []
        for m in markets:
            annotated = _annotate_slug(m)
            for token in m["tokens"]:
                subscriptions.append(token["token_id"])
                slug_map[token["token_id"]] = annotated

        # Provide live order-book + slug context so shocks classify richly.
        def provider(token_id: str) -> MarketContext:
            try:
                book = client.get_orderbook(token_id)
                bids = book.bids
            except Exception:
                bids = []
            return MarketContext(slug=slug_map.get(token_id, ""), bids=bids, goal_diff=0)

        detector = LiveDetector(dists, on_shock, context_provider=provider)

        # Adapt the Polymarket client to the detector's run() expectations.
        async def _run_poly():
            async def cb(trade):
                # Advance existing positions (fills / take-profit / stop-loss)
                # before detecting new shocks on this trade.
                if paper:
                    engine.update(trade.market_id, trade.price, trade.timestamp_ms)
                else:
                    engine.check_fills()
                    engine.cancel_expired()
                    engine.check_stops(trade.market_id, trade.price)
                await detector.feed_trade(trade)

            await client.stream_trades(subscriptions, cb, slug_map=slug_map)

        logger.info(
            "Starting {} detector on {} tokens", "paper" if paper else "LIVE", len(subscriptions)
        )
        try:
            asyncio.run(_run_poly())
        except KeyboardInterrupt:
            logger.info("Shutting down")
        finally:
            if paper:
                logger.info("Paper summary: {}", engine.summary())
            client.close()
            storage.close()
        return 0

    logger.info(
        "Starting {} detector on {} markets", "paper" if paper else "LIVE", len(subscriptions)
    )
    try:
        asyncio.run(detector.run(subscriptions))
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        if paper:
            logger.info("Paper summary: {}", engine.summary())
        client.close()
        storage.close()
    return 0


def _exists(path: str) -> bool:
    import os

    return os.path.exists(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FIFA World Cup shock trading bot")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["ingest", "build", "backtest", "gridsearch", "paper", "live"],
        help="Operating mode",
    )
    parser.add_argument(
        "--source",
        default=config.DATA_SOURCE,
        choices=["polymarket", "dome"],
        help="Data source (default: polymarket)",
    )
    parser.add_argument("--db", default=config.DB_PATH, help="Path to DuckDB store")
    parser.add_argument(
        "--dist", default=config.DISTRIBUTIONS_PATH, help="Path to distributions.json (input)"
    )
    parser.add_argument(
        "--out", default=config.DISTRIBUTIONS_PATH, help="Output distributions path (build mode)"
    )
    # Polymarket discovery options.
    parser.add_argument(
        "--leagues",
        nargs="*",
        default=config.DEFAULT_LEAGUES,
        help=(
            "Soccer league tag slugs to discover via numeric tag id "
            "(default: curated popular leagues). Pass with no values to disable "
            "and fall back to --tag-slug."
        ),
    )
    parser.add_argument(
        "--tag-slug",
        dest="tag_slug",
        default="soccer",
        help="Single Gamma tag slug fallback when --leagues is empty (default: soccer)",
    )
    parser.add_argument(
        "--max-markets",
        dest="max_markets",
        type=int,
        default=None,
        help="Cap the number of markets discovered (polymarket)",
    )
    parser.add_argument(
        "--max-trades",
        dest="max_trades",
        type=int,
        default=None,
        help="Cap trades fetched per market (polymarket)",
    )
    parser.add_argument(
        "--all-market-types",
        dest="all_market_types",
        action="store_true",
        help=(
            "Disable the moneyline / 3-way validation filter and keep every "
            "market type (spreads, totals, player props, etc.)"
        ),
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help=(
            "When the Data API 3500-trade cap truncates a market, backfill the "
            "older window from 1-minute prices-history (polymarket ingest)"
        ),
    )
    # Strategy / execution options.
    parser.add_argument(
        "--buckets",
        nargs="*",
        default=None,
        help=(
            "Only trade shocks whose bucket key (league|favoritism|depth|time|goal) "
            "contains one of these substrings, e.g. --buckets moderate_fav late. "
            "Applies to backtest, gridsearch, paper and live."
        ),
    )
    parser.add_argument(
        "--tp-grid",
        dest="tp_grid",
        nargs="*",
        type=float,
        default=[0.02, 0.03, 0.04, 0.05, 0.06],
        help="Take-profit values (in price units) to sweep in gridsearch mode",
    )
    parser.add_argument(
        "--sl-grid",
        dest="sl_grid",
        nargs="*",
        type=float,
        default=[0.0, 0.06, 0.08, 0.10, 0.15],
        help="Stop-loss values to sweep in gridsearch mode (0 disables the stop)",
    )
    parser.add_argument(
        "--hold-grid",
        dest="hold_grid",
        nargs="*",
        type=float,
        default=[0, 60, 180, 300, 600],
        help=(
            "Max-hold seconds to sweep in gridsearch mode (0 disables the timed "
            "exit, holding to end-of-data settlement)"
        ),
    )
    # Dome discovery options (used only with --source dome).
    parser.add_argument(
        "--tags", nargs="*", default=["Soccer"], help="Dome market tags to discover"
    )
    parser.add_argument(
        "--search", default=None, help="Dome fuzzy market search string (overrides --tags)"
    )
    parser.add_argument(
        "--status", default="closed", help="Market status filter (default: closed)"
    )
    parser.add_argument(
        "--start", type=int, default=None, help="Dome ingest start time (Unix seconds)"
    )
    parser.add_argument(
        "--end", type=int, default=None, help="Dome ingest end time (Unix seconds)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dispatch = {
        "ingest": cmd_ingest,
        "build": cmd_build,
        "backtest": cmd_backtest,
        "gridsearch": cmd_gridsearch,
        "paper": cmd_paper,
        "live": cmd_live,
    }
    return dispatch[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
