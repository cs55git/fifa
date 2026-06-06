"""Tests for the paper trading engine."""

from __future__ import annotations

from data.models import ShockEvent
from execution.paper import PaperEngine


def _shock(market_id="m1", bucket="deep|moderate_fav|balanced|mid|level", detected_at_ms=0):
    return ShockEvent(
        market_id=market_id,
        slug="epl-game",
        peak=0.30,
        floor=0.10,
        depth=0.20,
        pre_price=0.30,
        elapsed_ms=35 * 60_000,
        goal_diff=0,
        bids=[],
        bucket_key=bucket,
        detected_at_ms=detected_at_ms,
    )


def test_paper_fill_simulation():
    eng = PaperEngine()
    eng.submit(_shock(), [{"percentile": 75, "price": 0.18, "size": 10.0}])
    eng.update("m1", current_price=0.17, current_ts_ms=5_000)
    order = eng.orders[0]
    assert order["status"] == "filled"
    assert order["fill_price"] == 0.18
    # 10 dollars / 0.18 ~= 55.5 shares
    assert abs(order["shares"] - (10.0 / 0.18)) < 1e-9


def test_paper_exit_simulation():
    eng = PaperEngine(exit_cents=0.04, stop_loss_cents=None)
    eng.submit(_shock(), [{"percentile": 90, "price": 0.14, "size": 15.0}])
    eng.update("m1", current_price=0.14, current_ts_ms=1_000)  # fill at 0.14
    eng.update("m1", current_price=0.18, current_ts_ms=2_000)  # recovers +4c -> exit
    order = eng.orders[0]
    assert order["status"] == "exited"
    assert order["exit_price"] == 0.18
    shares = 15.0 / 0.14
    assert abs(order["pnl"] - shares * (0.18 - 0.14)) < 1e-9


def test_paper_expiry():
    eng = PaperEngine()
    eng.submit(_shock(detected_at_ms=0), [{"percentile": 50, "price": 0.22, "size": 5.0}])
    # 65s later with price still above the limit -> expired (not filled).
    eng.update("m1", current_price=0.30, current_ts_ms=65_000)
    assert eng.orders[0]["status"] == "expired"


def test_paper_no_fill_when_price_above_limit():
    eng = PaperEngine()
    eng.submit(_shock(), [{"percentile": 50, "price": 0.22, "size": 5.0}])
    eng.update("m1", current_price=0.25, current_ts_ms=5_000)
    assert eng.orders[0]["status"] == "open"


def test_paper_summary_win_rate():
    eng = PaperEngine(exit_cents=0.04, stop_loss_cents=None)
    # 10 separate shocks, each one order that fills and exits profitably.
    for i in range(10):
        s = _shock(market_id=f"m{i}", detected_at_ms=0)
        eng.submit(s, [{"percentile": 90, "price": 0.14, "size": 15.0}])
        eng.update(f"m{i}", current_price=0.14, current_ts_ms=1_000)
        eng.update(f"m{i}", current_price=0.18, current_ts_ms=2_000)

    summary = eng.summary()
    assert summary["total_orders"] == 10
    assert summary["fill_rate"] == 1.0
    assert summary["win_rate"] == 1.0
    assert summary["total_pnl"] > 0


def test_paper_summary_mixed():
    eng = PaperEngine(exit_cents=0.04, stop_loss_cents=None)
    # One fills+exits, one expires.
    eng.submit(_shock(market_id="a"), [{"percentile": 90, "price": 0.14, "size": 15.0}])
    eng.update("a", current_price=0.14, current_ts_ms=1_000)
    eng.update("a", current_price=0.18, current_ts_ms=2_000)

    eng.submit(_shock(market_id="b"), [{"percentile": 50, "price": 0.22, "size": 5.0}])
    eng.update("b", current_price=0.30, current_ts_ms=65_000)  # expires

    summary = eng.summary()
    assert summary["total_orders"] == 2
    assert summary["fill_rate"] == 0.5
    assert summary["win_rate"] == 1.0  # of exited orders, all won


def test_paper_stop_loss_cuts_position():
    """A filled position breaching the stop loss exits at the current price."""

    eng = PaperEngine(stop_loss_cents=0.08)
    eng.submit(_shock(market_id="m1"), [{"percentile": 90, "price": 0.20, "size": 10.0}])
    eng.update("m1", current_price=0.20, current_ts_ms=1_000)  # fill at 0.20
    eng.update("m1", current_price=0.11, current_ts_ms=2_000)  # <= 0.12 stop -> cut
    order = eng.orders[0]
    assert order["status"] == "stopped"
    assert order["exit_price"] == 0.11
    shares = 10.0 / 0.20
    assert abs(order["pnl"] - shares * (0.11 - 0.20)) < 1e-9
    assert order["pnl"] < 0


def test_paper_take_profit_precedes_stop_loss():
    """A bounce to the take-profit wins even with a stop loss configured."""

    eng = PaperEngine(exit_cents=0.04, stop_loss_cents=0.08)
    eng.submit(_shock(market_id="m1"), [{"percentile": 90, "price": 0.14, "size": 15.0}])
    eng.update("m1", current_price=0.14, current_ts_ms=1_000)
    eng.update("m1", current_price=0.18, current_ts_ms=2_000)  # +4c target
    assert eng.orders[0]["status"] == "exited"


def test_paper_stop_loss_disabled():
    """With stop loss disabled, a falling position rides to settlement instead."""

    eng = PaperEngine(stop_loss_cents=None)
    eng.submit(_shock(market_id="m1"), [{"percentile": 90, "price": 0.20, "size": 10.0}])
    eng.update("m1", current_price=0.20, current_ts_ms=1_000)
    eng.update("m1", current_price=0.05, current_ts_ms=2_000)  # would breach a stop
    assert eng.orders[0]["status"] == "filled"  # not stopped


def test_paper_max_hold_exits_at_market():
    """A position held past max_hold closes at the prevailing price."""

    eng = PaperEngine(exit_cents=0.10, stop_loss_cents=None, max_hold_seconds=300)
    eng.submit(_shock(market_id="m1"), [{"percentile": 90, "price": 0.20, "size": 10.0}])
    eng.update("m1", current_price=0.20, current_ts_ms=1_000)  # fill at t=1s
    # 200s later: neither TP (0.30) nor hold window reached -> still filled.
    eng.update("m1", current_price=0.23, current_ts_ms=201_000)
    assert eng.orders[0]["status"] == "filled"
    # 301s after fill: past the 300s hold window -> timed exit at market (0.23).
    eng.update("m1", current_price=0.23, current_ts_ms=302_000)
    order = eng.orders[0]
    assert order["status"] == "timed"
    assert order["exit_price"] == 0.23
    shares = 10.0 / 0.20
    assert abs(order["pnl"] - shares * (0.23 - 0.20)) < 1e-9


def test_paper_take_profit_precedes_max_hold():
    """Take profit still wins even if the hold window has elapsed."""

    eng = PaperEngine(exit_cents=0.04, stop_loss_cents=None, max_hold_seconds=60)
    eng.submit(_shock(market_id="m1"), [{"percentile": 90, "price": 0.14, "size": 15.0}])
    eng.update("m1", current_price=0.14, current_ts_ms=1_000)
    # Past hold window AND at the TP target -> TP takes priority (win).
    eng.update("m1", current_price=0.18, current_ts_ms=120_000)
    assert eng.orders[0]["status"] == "exited"


def test_paper_settle_realizes_loss():
    """A filled position that never recovers is a real loss after settlement."""

    eng = PaperEngine(exit_cents=0.04, stop_loss_cents=None)
    eng.submit(_shock(market_id="m1"), [{"percentile": 90, "price": 0.14, "size": 15.0}])
    eng.update("m1", current_price=0.14, current_ts_ms=1_000)  # fill at 0.14
    # Price keeps dropping and never recovers; market data ends at 0.03.
    eng.update("m1", current_price=0.08, current_ts_ms=2_000)
    eng.settle("m1", final_price=0.03, final_ts_ms=3_000)

    order = eng.orders[0]
    assert order["status"] == "settled"
    assert order["pnl"] < 0  # bought at 0.14, marked to 0.03


def test_paper_summary_win_rate_not_perfect_after_settlement():
    """Settlement turns un-recovered positions into losses, so win rate < 1."""

    eng = PaperEngine(exit_cents=0.04, stop_loss_cents=None)
    # Winner: fills then recovers +4c.
    eng.submit(_shock(market_id="win"), [{"percentile": 90, "price": 0.14, "size": 15.0}])
    eng.update("win", current_price=0.14, current_ts_ms=1_000)
    eng.update("win", current_price=0.18, current_ts_ms=2_000)
    # Loser: fills then collapses to 0.02 and gets settled.
    eng.submit(_shock(market_id="lose"), [{"percentile": 90, "price": 0.14, "size": 15.0}])
    eng.update("lose", current_price=0.14, current_ts_ms=1_000)
    eng.settle("lose", final_price=0.02, final_ts_ms=3_000)

    summary = eng.summary()
    assert summary["win_rate"] == 0.5
    assert summary["fill_rate"] == 1.0
