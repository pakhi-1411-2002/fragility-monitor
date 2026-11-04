"""Central settings for the Fragility Monitor.

Every number that might change lives here. Other modules import from this
file and never hard-code values. Log any change to a threshold in the
master plan's change log (Section 13).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# This file sits at <repo>/src/fragility/config.py, so the repo root is
# three levels up (config.py -> fragility -> src -> repo root).
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
DEMO_DIR = DATA_DIR / "demo"

ETF_UNIVERSE_CSV = REFERENCE_DIR / "etf_universe.csv"
ETF_HOLDINGS_CSV = REFERENCE_DIR / "etf_holdings.csv"
SCENARIOS_CSV = REFERENCE_DIR / "scenarios.csv"
NAME_MAP_CSV = REFERENCE_DIR / "name_map.csv"
PRICE_CACHE = CACHE_DIR / "prices.parquet"
HOLDINGS_SOURCES_CSV = REFERENCE_DIR / "holdings_sources.csv"

# ---------------------------------------------------------------------------
# Prices (Section 5.3)
# ---------------------------------------------------------------------------
START_DATE = "2015-01-01"
BASE_CURRENCY = "GBP"
FX_TICKER_GBPUSD = "GBPUSD=X"       # used to convert USD prices to GBP
PRICE_CACHE_MAX_AGE_DAYS = 1        # local: refresh if cache older than this
PRICE_CACHE_MAX_AGE_DAYS_DEPLOYED = 7
FFILL_LIMIT_DAYS = 5                # fill holiday gaps up to this many days
MAX_MISSING_SHARE = 0.10            # drop a ticker if >10% of days missing

# ---------------------------------------------------------------------------
# Holdings and look-through (Sections 5.2 and 6.1)
# ---------------------------------------------------------------------------
TOP_N_HOLDINGS = 500
SHARE_PRICE_TOP_N = 50                 # keep top 50 per ETF, rest -> residual
RESIDUAL_LABEL = "Other (residual)"
WEIGHT_SUM_TOLERANCE = 0.005        # ETF weights must sum to 1 +/- this

# ---------------------------------------------------------------------------
# Correlation (Section 6.3)
# ---------------------------------------------------------------------------
MARKET_TICKER = "VWRL.L"            # defines "bad market weeks"
CORR_WINDOW_YEARS = 3
CORR_FREQ = "W-FRI"                 # weekly returns, week ending Friday
STRESS_WEEK_THRESHOLD = -0.02       # market fell more than 2% that week
CLUSTER_CUT = 0.30                  # distance cut for "moves together" groups
MIN_HISTORY_YEARS = 2               # flag holdings with less history

# ---------------------------------------------------------------------------
# Stress tests (Section 7.1)
# ---------------------------------------------------------------------------
HYPOTHETICAL_SHOCKS = [
    {"name": "US tech -20%", "sector": "Information Technology",
     "country": "United States", "shock": -0.20},
    {"name": "UK equities -15%", "sector": None,
     "country": "United Kingdom", "shock": -0.15},
]

# ---------------------------------------------------------------------------
# Fragility score (Section 7.2) - PLACEHOLDERS until Day 7 calibration
# ---------------------------------------------------------------------------
# Each metric maps to 0-100 on a straight line from "calm" (0) to "alarm" (100).
SCORE_COMPONENTS = {
    "company_concentration": {"calm": 0.15, "alarm": 0.60, "weight": 0.30},  # top-10 share
    "sector_country_tilt":   {"calm": 0.25, "alarm": 0.70, "weight": 0.20},  # max share
    "moves_together":        {"calm": 0.30, "alarm": 0.90, "weight": 0.25},  # stress corr
    "crash_loss":            {"calm": 0.10, "alarm": 0.40, "weight": 0.25},  # worst loss
}
COUNTRY_SHARE_DIVISOR = 1.5         # some home/US bias is normal

SCORE_BANDS = [                     # (upper limit, label)
    (33, "Robust"),
    (66, "Moderate"),
    (100, "Fragile"),
]

# ---------------------------------------------------------------------------
# Alerts (Section 8) - PLACEHOLDERS
# ---------------------------------------------------------------------------
ALERT_THRESHOLDS = {
    "A1_single_company": 0.08,      # one company >= 8% ...
    "A1_min_routes": 2,             # ... reached through 2+ routes
    "A2_top10_share": 0.40,
    "A3_fund_overlap": 0.70,
    "A4_sector_share": 0.35,
    "A5_country_share": 0.75,
    "A6_stress_corr": 0.80,
    "A7_crash_loss": 0.30,
    "A8_min_holdings": 5,
    "A8_max_effective_groups": 2,
    "P1_max_score": 33,
}
SEVERITY_HIGH_MULTIPLIER = 1.5      # metric >= 1.5x threshold -> severity 3
MAX_ALERTS_SHOWN = 3

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
PERF_BUDGET_SECONDS = 10
DISCLAIMER = "Educational tool, not financial advice."

# ---------------------------------------------------------------------------
# Safety check: score weights must add up to 1
# ---------------------------------------------------------------------------
_total_weight = sum(c["weight"] for c in SCORE_COMPONENTS.values())
assert abs(_total_weight - 1.0) < 1e-9, f"Score weights sum to {_total_weight}, not 1"