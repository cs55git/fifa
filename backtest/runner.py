"""End-to-end backtester.

Replays stored historical trades through the full pipeline -- detection,
classification, distribution lookup, ladder construction -- and simulates fills
and exits using the same :class:`PaperEngine` used in paper trading. Produces a
per-bucket performance report so you can find the segments worth trading (the
article highlights ``moderate_fav`` as an example).

Simplifying assumption: fills are evaluated on trades that arrive *after* the
shock is detected, so a shallow order that would have filled on the shock's own
floor tick is treated conservatively as unfilled. This biases results slightly
pessimistic, which is the safe direction for a trading system.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Optional

import pandas as pd
from loguru import logger

import config
from classifier.bucket import bucket_key, bucket_matches
from classifier.shock import detect_shock
from data.models import ShockEvent, Trade
from detector.window import SlidingWindow
from distributions import store
from execution.ladder import build_ladder
from execution.paper import PaperEngine

Level = tuple[float, float]


class BacktestContext:
    def __init__(
        self,
        slug: str = "",
        bids: Optional[Sequence[Level]] = None,
        goal_diff: int = 0,
        kickoff_ms: Optional[int] = None,
    ) -> None:
        self.slug = slug
        self.bids = list(bids) if bids else []
        self.goal_diff = goal_diff
        self.kickoff_ms = kickoff_ms


ContextFn = Callable[[dict, Sequence[Trade]], BacktestContext]


class BacktestRunner:
    def __init__(
        self,
        storage,
        distributions: dict,
        *,
        capital: float = config.CAPITAL_PER_SHOCK,
        context_fn: Optional[ContextFn] = None,
        exit_cents: float = config.EXIT_CENTS,
        stop_loss_cents: Optional[float] = config.STOP_LOSS_CENTS,
        max_hold_seconds: Optional[float] = config.MAX_HOLD_SECONDS,
        bucket_filter: Optional[Sequence[str]] = None,
    ) -> None:
        self.storage = storage
        self.distributions = distributions
        self.capital = capital
        self.context_fn = context_fn
        self.bucket_filter = list(bucket_filter) if bucket_filter else None
        self.engine = PaperEngine(
            exit_cents=exit_cents,
            stop_loss_cents=stop_loss_cents,
            max_hold_seconds=max_hold_seconds,
        )
        self.results: pd.DataFrame = pd.DataFrame()
        self.n_shocks = 0

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, market_ids: Optional[Sequence[str]] = None) -> pd.DataFrame:
        if market_ids is None:
            market_ids = self.storage.list_market_ids()
        for market_id in market_ids:
            trades = self.storage.get_trades(market_id)
            self._run_market(market_id, trades)
        self.results = self._to_frame()
        return self.results

    def run_trades(self, trades: Sequence[Trade]) -> pd.DataFrame:
        """Run the backtest directly on an in-memory trade list (for tests)."""

        by_market: dict[str, list[Trade]] = {}
        for t in trades:
            by_market.setdefault(t.market_id, []).append(t)
        for market_id, mtrades in by_market.items():
            self._run_market(market_id, mtrades)
        self.results = self._to_frame()
        return self.results

    def _run_market(self, market_id: str, trades: Sequence[Trade]) -> None:
        if not trades:
            return
        ordered = sorted(trades, key=lambda t: t.timestamp_ms)
        window = SlidingWindow()
        cooldown: dict[str, int] = {}

        for trade in ordered:
            # Advance existing orders with this price before detecting new shocks.
            self.engine.update(market_id, trade.price, trade.timestamp_ms)

            window.add(trade)
            shock = detect_shock(window.trades, cooldown)
            if shock is None:
                continue

            ctx = (
                self.context_fn(shock, window.trades)
                if self.context_fn
                else BacktestContext(slug=trade.slug)
            )
            kickoff = ctx.kickoff_ms if ctx.kickoff_ms is not None else ordered[0].timestamp_ms
            elapsed_ms = max(0, shock["detected_at_ms"] - kickoff)
            slug = ctx.slug or trade.slug
            key = bucket_key(slug, shock["peak"], ctx.bids, elapsed_ms, ctx.goal_diff)

            if not bucket_matches(key, self.bucket_filter):
                continue

            percentiles = store.lookup(key, self.distributions)
            orders = build_ladder(shock["peak"], percentiles, self.capital)
            if not orders:
                continue

            event = ShockEvent(
                market_id=market_id,
                slug=slug,
                peak=shock["peak"],
                floor=shock["floor"],
                depth=shock["depth"],
                pre_price=shock["peak"],
                elapsed_ms=elapsed_ms,
                goal_diff=ctx.goal_diff,
                bids=list(ctx.bids),
                bucket_key=key,
                detected_at_ms=shock["detected_at_ms"],
            )
            self.engine.submit(event, orders)
            self.n_shocks += 1

        # Settle any positions still open at the end of this market's data so the
        # backtest realizes losses (not just the +EXIT_CENTS winners).
        last = ordered[-1]
        self.engine.settle(market_id, last.price, last.timestamp_ms)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def _to_frame(self) -> pd.DataFrame:
        rows = []
        realized = {"filled", "exited", "stopped", "timed", "settled"}
        for o in self.engine.orders:
            duration = None
            if o["status"] in realized and o["updated_at_ms"] is not None:
                duration = o["updated_at_ms"] - o["created_at_ms"]
            rows.append(
                {
                    "bucket_key": o["bucket_key"],
                    "market_id": o["market_id"],
                    "percentile": o["percentile"],
                    "status": o["status"],
                    "fill_price": o["fill_price"],
                    "exit_price": o["exit_price"],
                    "pnl": o["pnl"],
                    "duration_ms": duration,
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "bucket_key",
                "market_id",
                "percentile",
                "status",
                "fill_price",
                "exit_price",
                "pnl",
                "duration_ms",
            ],
        )

    def bucket_summary(self) -> pd.DataFrame:
        """Aggregate results per bucket: order count, fill/win rate, PnL."""

        df = self.results if not self.results.empty else self._to_frame()
        if df.empty:
            return pd.DataFrame(
                columns=["bucket_key", "orders", "fills", "exits", "win_rate", "fill_rate", "total_pnl", "avg_pnl"]
            )

        records = []
        for key, g in df.groupby("bucket_key"):
            orders = len(g)
            realized_statuses = ["exited", "stopped", "timed", "settled"]
            fills = int((g["status"].isin(["filled"] + realized_statuses)).sum())
            # Realized exits are take-profit hits ("exited"), stop-loss cuts
            # ("stopped"), max-hold market exits ("timed") and mark-to-market
            # settlements ("settled"); wins are realized exits with positive PnL.
            exited = g[g["status"].isin(realized_statuses)]
            n_exit = len(exited)
            wins = int((exited["pnl"] > 0).sum())
            total_pnl = float(exited["pnl"].sum())
            records.append(
                {
                    "bucket_key": key,
                    "orders": orders,
                    "fills": fills,
                    "exits": n_exit,
                    "win_rate": (wins / n_exit) if n_exit else 0.0,
                    "fill_rate": (fills / orders) if orders else 0.0,
                    "total_pnl": total_pnl,
                    "avg_pnl": (total_pnl / n_exit) if n_exit else 0.0,
                }
            )
        return pd.DataFrame(records).sort_values("total_pnl", ascending=False).reset_index(drop=True)

    def report(self, top_n: int = 100) -> pd.DataFrame:
        summary = self.bucket_summary()
        total_pnl = float(self.results["pnl"].fillna(0).sum()) if not self.results.empty else 0.0
        logger.info("Backtest complete: {} shocks, {} orders", self.n_shocks, len(self.results))
        logger.info("Total PnL: {:.2f}", total_pnl)
        if not summary.empty:
            top = summary.sort_values("win_rate", ascending=False).head(top_n)
            logger.info("Top {} buckets by win rate:\n{}", top_n, top.to_string(index=False))
        return summary
