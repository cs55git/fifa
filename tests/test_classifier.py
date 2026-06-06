"""Tests for shock detection and the five-dimension classifiers."""

from __future__ import annotations

from classifier.bucket import bucket_key, bucket_matches
from classifier.dimensions import (
    classify_depth,
    classify_favoritism,
    classify_goal,
    classify_league,
    classify_time,
)
from classifier.shock import detect_shock
from data.models import Trade


def _window(prices, market_id="m1", slug="epl-game", start_ms=0, step_ms=10_000):
    return [
        Trade(
            market_id=market_id,
            slug=slug,
            timestamp_ms=start_ms + i * step_ms,
            price=p,
            size=1.0,
        )
        for i, p in enumerate(prices)
    ]


# ---------------------------------------------------------------------------
# Shock detection
# ---------------------------------------------------------------------------
def test_detect_shock_fires():
    # 0.50 -> 0.35 = 30% drop, 15c absolute. Both thresholds exceeded.
    trades = _window([0.50, 0.45, 0.35])
    shock = detect_shock(trades, {})
    assert shock is not None
    assert shock["peak"] == 0.50
    assert shock["floor"] == 0.35
    assert abs(shock["depth"] - 0.15) < 1e-9


def test_detect_shock_pct_threshold():
    # 0.50 -> 0.44 = 12% drop, below 15% threshold.
    trades = _window([0.50, 0.44])
    assert detect_shock(trades, {}) is None


def test_detect_shock_abs_threshold():
    # 0.10 -> 0.085 = 15% drop but only 1.5c absolute, below 8c threshold.
    trades = _window([0.10, 0.085])
    assert detect_shock(trades, {}) is None


def test_detect_shock_needs_two_trades():
    assert detect_shock(_window([0.5]), {}) is None
    assert detect_shock([], {}) is None


def test_cooldown_blocks():
    tracker: dict[str, int] = {}
    first = detect_shock(_window([0.50, 0.35], start_ms=0), tracker)
    assert first is not None
    # Second shock 60s later (within the 180s cooldown) is blocked.
    second = detect_shock(_window([0.50, 0.35], start_ms=60_000), tracker)
    assert second is None
    # A shock well past the cooldown fires again.
    third = detect_shock(_window([0.50, 0.35], start_ms=200_000), tracker)
    assert third is not None


# ---------------------------------------------------------------------------
# League
# ---------------------------------------------------------------------------
def test_classify_league_deep():
    assert classify_league("epl-man-utd-vs-arsenal") == "deep"
    assert classify_league("uefa-champions-league-final") == "deep"
    assert classify_league("fifa-world-cup-bra-vs-arg") == "deep"


def test_classify_league_thin_and_unknown():
    assert classify_league("mls-some-game") == "thin"
    assert classify_league("obscure-friendly") == "unknown"
    assert classify_league("") == "unknown"


def test_classify_league_from_annotated_slug():
    # Ingestion annotates soccer slugs as "<league-slug>/<market-slug>".
    assert classify_league("premier-league/epl-mun-vs-ars") == "deep"
    assert classify_league("la-liga/rma-vs-fcb-2026") == "deep"
    assert classify_league("uefa-champions-league/final-2026") == "deep"
    assert classify_league("fifa-world-cup/bra-vs-arg") == "deep"
    # Minor leagues -> thin.
    assert classify_league("norway-eliteserien/nor-sar-bog-2025-10-18-sar") == "thin"
    assert classify_league("primeira-liga/por-spo-vs-ben") == "thin"


# ---------------------------------------------------------------------------
# Favoritism
# ---------------------------------------------------------------------------
def test_classify_favoritism_all_bands():
    assert classify_favoritism(0.90) == "heavy_fav"
    assert classify_favoritism(0.85) == "heavy_fav"  # >= 0.85 is heavy
    assert classify_favoritism(0.80) == "moderate_fav"
    assert classify_favoritism(0.70) == "slight_fav"
    assert classify_favoritism(0.50) == "balanced"
    assert classify_favoritism(0.30) == "underdog"


# ---------------------------------------------------------------------------
# Depth
# ---------------------------------------------------------------------------
def test_classify_depth_top_heavy():
    # Top 3 levels hold 80 of 100 total size -> top_heavy.
    bids = [(0.50, 40), (0.49, 25), (0.48, 15), (0.47, 10), (0.46, 10)]
    assert classify_depth(bids) == "top_heavy"


def test_classify_depth_balanced_and_deep():
    # Top 3 = 60% -> balanced.
    balanced = [(0.50, 20), (0.49, 20), (0.48, 20), (0.47, 20), (0.46, 20)]
    assert classify_depth(balanced) == "balanced"
    # Top 3 small relative to a deep book -> deep_liq.
    deep = [(0.50, 5), (0.49, 5), (0.48, 5), (0.47, 40), (0.46, 45)]
    assert classify_depth(deep) == "deep_liq"


def test_classify_depth_empty():
    assert classify_depth([]) == "unknown"


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------
def test_classify_time_all_bands():
    assert classify_time(0) == "early"
    assert classify_time(20 * 60_000) == "mid"
    assert classify_time(65 * 60_000) == "late"
    assert classify_time(85 * 60_000) == "final"


# ---------------------------------------------------------------------------
# Goal state
# ---------------------------------------------------------------------------
def test_classify_goal_all_bands():
    assert classify_goal(0) == "level"
    assert classify_goal(1) == "close"
    assert classify_goal(2) == "comfortable"
    assert classify_goal(3) == "blowout"
    assert classify_goal(-2) == "comfortable"  # absolute difference


# ---------------------------------------------------------------------------
# Bucket key
# ---------------------------------------------------------------------------
def test_bucket_key_format():
    key = bucket_key("epl-game", 0.30, [(0.50, 20), (0.49, 20), (0.48, 20)], 35 * 60_000, 0)
    parts = key.split("|")
    assert len(parts) == 5


def test_bucket_key_example():
    # Major league, underdog, balanced book, mid match, level score.
    bids = [(0.50, 20), (0.49, 20), (0.48, 20), (0.47, 20), (0.46, 20)]
    key = bucket_key("epl-man-utd", 0.30, bids, 35 * 60_000, 0)
    assert key == "deep|underdog|balanced|mid|level"


def test_bucket_matches_filter():
    key = "deep|moderate_fav|balanced|late|level"
    # No filter -> everything passes.
    assert bucket_matches(key, None) is True
    assert bucket_matches(key, []) is True
    # Single dimension substring.
    assert bucket_matches(key, ["moderate_fav"]) is True
    assert bucket_matches(key, ["late"]) is True
    # Any-of semantics.
    assert bucket_matches(key, ["heavy_fav", "late"]) is True
    # No match.
    assert bucket_matches(key, ["heavy_fav"]) is False
    # Whole-key match.
    assert bucket_matches(key, [key]) is True
