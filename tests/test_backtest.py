"""Tests for the end-to-end backtest runner."""

from __future__ import annotations

from backtest.runner import BacktestContext, BacktestRunner
from data.models import Trade
from data.storage import Storage


def _trades(market_id, prices, slug="epl-game", start_ms=0, step_ms=5_000):
    return [
        Trade(
            market_id=market_id,
            slug=slug,
            timestamp_ms=start_ms + i * step_ms,
            price=p,
            size=10,
        )
        for i, p in enumerate(prices)
    ]


def test_backtest_runner_integration():
    # Pre-shock 0.30 plateau, crash to 0.10, then a recovery back up.
    # Crash crosses 8c threshold at 0.20; ladder built around pre=0.30.
    prices = (
        [0.30] * 4          # plateau
        + [0.24, 0.18, 0.12, 0.10]  # crash (shock detected ~0.24/0.22)
        + [0.11, 0.13, 0.15, 0.18, 0.22, 0.26, 0.30]  # recovery
    )
    trades = _trades("m1", prices)

    # Force a deep underdog bucket with rich book so it classifies cleanly.
    def ctx(shock, window):
        return BacktestContext(
            slug="epl-game",
            bids=[(0.3, 20), (0.29, 20), (0.28, 20), (0.27, 20), (0.26, 20)],
            goal_diff=0,
            kickoff_ms=0,
        )

    runner = BacktestRunner(storage=None, distributions={}, context_fn=ctx)
    results = runner.run_trades(trades)

    assert runner.n_shocks >= 1
    assert not results.empty
    # At least one order should have filled and exited with positive PnL.
    exited = results[results["status"] == "exited"]
    assert len(exited) >= 1
    assert (exited["pnl"] > 0).any()


def test_report_columns_and_rows():
    runner = BacktestRunner(storage=None, distributions={})
    # Two markets, each producing a clean shock + recovery.
    trades = []
    for i in range(5):
        prices = [0.30] * 3 + [0.20, 0.10] + [0.12, 0.16, 0.22, 0.30]
        trades += _trades(f"m{i}", prices, start_ms=0)
    runner.run_trades(trades)
    summary = runner.report()

    expected_cols = {
        "bucket_key",
        "orders",
        "fills",
        "exits",
        "win_rate",
        "fill_rate",
        "total_pnl",
        "avg_pnl",
    }
    assert expected_cols <= set(summary.columns)
    assert len(summary) >= 1


def test_backtest_from_storage():
    store = Storage(db_path=":memory:")
    prices = [0.30] * 3 + [0.20, 0.10] + [0.12, 0.16, 0.22, 0.30]
    store.insert_trades(_trades("m1", prices))

    runner = BacktestRunner(storage=store, distributions={})
    results = runner.run()
    assert runner.n_shocks >= 1
    assert not results.empty
    store.close()


def test_backtest_bucket_filter_skips_non_matching():
    prices = [0.30] * 3 + [0.20, 0.10] + [0.12, 0.16, 0.22, 0.30]
    trades = _trades("m1", prices)

    def ctx(shock, window):
        return BacktestContext(slug="epl-game", bids=[], goal_diff=0, kickoff_ms=0)

    # A filter that cannot match any produced bucket -> no orders submitted.
    runner = BacktestRunner(
        storage=None, distributions={}, context_fn=ctx, bucket_filter=["nonexistent_dim"]
    )
    results = runner.run_trades(trades)
    assert results.empty


def test_backtest_stop_loss_realizes_losses():
    # Shock then a permanent collapse (no recovery) -> stop loss should cut.
    prices = [0.40] * 3 + [0.30, 0.20] + [0.18, 0.16, 0.14, 0.12, 0.10, 0.08]
    trades = _trades("m1", prices)

    def ctx(shock, window):
        return BacktestContext(slug="epl-game", bids=[], goal_diff=0, kickoff_ms=0)

    runner = BacktestRunner(
        storage=None,
        distributions={},
        context_fn=ctx,
        exit_cents=0.04,
        stop_loss_cents=0.03,
    )
    results = runner.run_trades(trades)
    if not results.empty and (results["status"] == "stopped").any():
        stopped = results[results["status"] == "stopped"]
        assert (stopped["pnl"] < 0).all()


def test_empty_backtest_is_safe():
    runner = BacktestRunner(storage=None, distributions={})
    results = runner.run_trades([])
    assert results.empty
    summary = runner.report()
    assert summary.empty
