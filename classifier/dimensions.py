"""The five independent shock classification dimensions.

Each shock is classified across five dimensions. The combination of the five
labels forms a bucket key that maps to a historical depth distribution. The exact
boundaries below are taken directly from the strategy article.
"""

from __future__ import annotations

from collections.abc import Sequence

import config

# A (price, size) order-book level.
Level = tuple[float, float]


# Slugs that identify deep, liquid major leagues / competitions.
MAJOR_LEAGUE_TOKENS = (
    "epl",
    "premier-league",
    "premier_league",
    "champions-league",
    "champions_league",
    "ucl",
    "la-liga",
    "la_liga",
    "laliga",
    "bundesliga",
    "serie-a",
    "serie_a",
    "seriea",
    "ligue-1",
    "ligue_1",
    "ligue1",
    "world-cup",
    "world_cup",
    "worldcup",
    "fifa",
)

# Slugs that identify thin, more erratic minor leagues.
MINOR_LEAGUE_TOKENS = (
    "mls",
    "eredivisie",
    "championship",
    "league-one",
    "league-two",
    "primeira",
    "super-lig",
    "j-league",
    "k-league",
    "a-league",
    "liga-mx",
    "brasileirao",
)


def classify_league(slug: str) -> str:
    """Classify the market's league tier from its (possibly annotated) slug.

    Polymarket soccer market slugs do not contain the league name, so the
    ingestion layer annotates the slug with the league slug it was discovered
    under (e.g. ``"premier-league/nor-sar-bog-..."``). We first consult the
    curated league registry (``config.SOCCER_LEAGUE_TIERS``) by substring match,
    then fall back to the generic major/minor token lists for raw slugs.

    Returns ``"deep"`` for major leagues, ``"thin"`` for minor leagues and
    ``"unknown"`` otherwise.
    """

    s = (slug or "").lower()
    # Longest league slugs first so e.g. "uefa-champions-league" wins over a
    # shorter accidental substring.
    for league_slug, tier in sorted(
        config.SOCCER_LEAGUE_TIERS.items(), key=lambda kv: len(kv[0]), reverse=True
    ):
        if league_slug in s:
            return tier
    if any(tok in s for tok in MAJOR_LEAGUE_TOKENS):
        return "deep"
    if any(tok in s for tok in MINOR_LEAGUE_TOKENS):
        return "thin"
    return "unknown"


def classify_favoritism(pre_price: float) -> str:
    """Classify how favored the team was just before the shock.

    ``pre_price`` is the implied probability (0-1).
    """

    if pre_price >= 0.85:
        return "heavy_fav"
    if pre_price >= 0.75:
        return "moderate_fav"
    if pre_price >= 0.60:
        return "slight_fav"
    if pre_price >= 0.45:
        return "balanced"
    return "underdog"


def classify_depth(bids: Sequence[Level]) -> str:
    """Classify order-book depth via the share of size in the top 3 bid levels.

    Top-heavy books (>=70% in top 3) have thin support below, so shocks go deep.
    Deep-liquidity books (<50% in top 3) have strong support and shallow shocks.
    """

    if not bids:
        return "unknown"

    total = sum(size for _, size in bids)
    if total <= 0:
        return "unknown"

    top_three = sorted(bids, key=lambda lvl: lvl[0], reverse=True)[:3]
    top_share = sum(size for _, size in top_three) / total

    if top_share >= 0.70:
        return "top_heavy"
    if top_share >= 0.50:
        return "balanced"
    return "deep_liq"


def classify_time(elapsed_ms: int) -> str:
    """Classify the match phase from elapsed match time in milliseconds."""

    minutes = elapsed_ms / 60_000
    if minutes < 15:
        return "early"
    if minutes < 60:
        return "mid"
    if minutes < 80:
        return "late"
    return "final"


def classify_goal(goal_diff: int) -> str:
    """Classify the score state from the absolute goal difference."""

    diff = abs(goal_diff)
    if diff == 0:
        return "level"
    if diff == 1:
        return "close"
    if diff == 2:
        return "comfortable"
    return "blowout"
