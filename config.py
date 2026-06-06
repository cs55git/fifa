"""Central configuration for the FIFA World Cup shock trading bot.

Secrets are read from environment variables (loaded from a local ``.env`` file
via ``python-dotenv``). Tuning parameters that govern the strategy live here as
module-level constants so they can be imported anywhere in the codebase.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Secrets / credentials (never hardcode these — always read from the env)
# ---------------------------------------------------------------------------
DOME_API_KEY = os.getenv("DOME_API_KEY", "")
POLYMARKET_API_KEY = os.getenv("POLYMARKET_API_KEY", "")
POLYMARKET_API_SECRET = os.getenv("POLYMARKET_API_SECRET", "")
POLYMARKET_API_PASSPHRASE = os.getenv("POLYMARKET_API_PASSPHRASE", "")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Default to paper mode unless explicitly overridden. Live trading must be an
# affirmative, conscious choice.
PAPER_MODE = _env_bool("PAPER_MODE", True)
CAPITAL_PER_SHOCK = _env_float("CAPITAL_PER_SHOCK", 50.0)


# ---------------------------------------------------------------------------
# Dome API endpoints (per https://docs.domeapi.io)
# REST base is https://api.domeapi.io/v1 ; the WebSocket connects to
# wss://ws.domeapi.io/<API_KEY> (the key is part of the URL path).
# Note: Dome announced end-of-life on 2026-04-28; if endpoints are retired,
# point these at Polymarket's successor API via the env vars.
# ---------------------------------------------------------------------------
DOME_BASE_URL = os.getenv("DOME_BASE_URL", "https://api.domeapi.io/v1")
DOME_WS_URL = os.getenv("DOME_WS_URL", "wss://ws.domeapi.io")


# ---------------------------------------------------------------------------
# Polymarket native API endpoints (public, no auth for reads)
#   Gamma  -> market/event discovery (slugs, conditionId, clobTokenIds)
#   Data   -> historical trade fills (/trades)
#   CLOB   -> order book (/book) and historical prices (/prices-history) + WS
# ---------------------------------------------------------------------------
POLYMARKET_GAMMA_URL = os.getenv("POLYMARKET_GAMMA_URL", "https://gamma-api.polymarket.com")
POLYMARKET_DATA_URL = os.getenv("POLYMARKET_DATA_URL", "https://data-api.polymarket.com")
POLYMARKET_CLOB_URL = os.getenv("POLYMARKET_CLOB_URL", "https://clob.polymarket.com")
POLYMARKET_CLOB_WS_URL = os.getenv(
    "POLYMARKET_CLOB_WS_URL", "wss://ws-subscriptions-clob.polymarket.com/ws/market"
)

# Which data source the ingestion / live layers default to: "polymarket" or "dome".
DATA_SOURCE = os.getenv("DATA_SOURCE", "polymarket")

# Soccer leagues to discover, mapped to their depth tier. Polymarket's /markets
# endpoint filters by numeric tag_id (resolved at runtime from these slugs via
# /tags/slug/<slug>); the generic "soccer" tag is too noisy. "deep" = major /
# liquid leagues with predictable recovery; "thin" = minor / erratic leagues.
SOCCER_LEAGUE_TIERS: dict[str, str] = {
    "premier-league": "deep",
    "la-liga": "deep",
    "bundesliga": "deep",
    "serie-a": "deep",
    "ligue-1": "deep",
    "uefa-champions-league": "deep",
    "champions-league": "deep",
    "fifa-world-cup": "deep",
    "world-cup": "deep",
    "mls": "thin",
    "norway-eliteserien": "thin",
    "primeira-liga": "thin",
}

# Default set of league slugs to discover during ingestion / live runs.
DEFAULT_LEAGUES: list[str] = list(SOCCER_LEAGUE_TIERS.keys())

# Discovery restricts to moneyline markets and validates the full 3-way match
# outcome set (team A win / team B win / draw) instead of using a slug blacklist.
# Set to None to query every sports market type.
SPORTS_MARKET_TYPE = "moneyline"


# ---------------------------------------------------------------------------
# Shock detection parameters
# ---------------------------------------------------------------------------
SHOCK_WINDOW_MS = 120_000  # 2 minute sliding window
SHOCK_DROP_PCT = 0.15  # minimum relative drop peak->floor
SHOCK_DROP_ABS = 0.08  # minimum absolute drop in cents (price units)
COOLDOWN_MS = 180_000  # 3 minute per-market cooldown


# ---------------------------------------------------------------------------
# Laddered execution parameters
# ---------------------------------------------------------------------------
# Capital weight allocated to each percentile band of the ladder.
LADDER_WEIGHTS: dict[int, float] = {50: 0, 75: 0, 90: 0.35, 95: 0.65}

# Take profit: fixed recovery target above the fill, sell back into the bounce.
EXIT_CENTS = 0.50
# Stop loss: cut a filled position when price falls this many cents below the
# fill (caps downside instead of riding it to resolution). Set to None to
# disable and rely solely on the take-profit / end-of-data settlement.
STOP_LOSS_CENTS: float | None = None
# Max hold: close a filled position at the prevailing market price after this
# many seconds, modelling the strategy's near-term bounce exit instead of
# holding to match resolution. Set to None to hold until end-of-data settlement.
MAX_HOLD_SECONDS: float | None = 60
ORDER_TTL_SECONDS = 60  # cancel unfilled orders after this many seconds
MIN_ORDER_PRICE = 0.01  # never place orders at or below 1 cent


# ---------------------------------------------------------------------------
# Distribution parameters
# ---------------------------------------------------------------------------
# Buckets with fewer than this many historical shocks are treated as untrusted
# and fall back to conservative shallow defaults.
MIN_BUCKET_SIZE = 5

# Percentile bands tracked for every bucket.
PERCENTILES: tuple[int, ...] = (50, 75, 90, 95)

# Conservative shallow defaults used for thin / unknown buckets.
FALLBACK_PERCENTILES: dict[int, float] = {50: 0.06, 75: 0.09, 90: 0.13, 95: 0.18}


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
DB_PATH = os.getenv("DB_PATH", "data_store.duckdb")
DISTRIBUTIONS_PATH = os.getenv("DISTRIBUTIONS_PATH", "distributions.json")
