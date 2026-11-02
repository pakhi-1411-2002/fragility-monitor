"""Tests for fragility.pipeline.

Part 1 builds a small made-up world (3 funds, synthetic prices) so the whole
pipeline runs offline in a fraction of a second.
Part 2 runs the real demo portfolios on your real data (needs the price cache).
"""

import io
import time

import numpy as np
import pandas as pd
import pytest

from fragility.config import DEMO_DIR, PRICE_CACHE
from fragility.pipeline import Report, analyse_csv, run_analysis
from fragility.portfolio import ShareInfo

IT, US, UK = "Information Technology", "United States", "United Kingdom"

# --- a small made-up world ----------------------------------------------------------
UNIVERSE = pd.DataFrame({
    "ticker": ["WORLD", "USTECH", "GILTS", "NEWBOND"],
    "name": ["World fund", "US tech fund", "UK gilts", "New bond fund"],
    "asset_class": ["equity", "equity", "bond", "bond"],
    "region": ["Global", "US", "UK", "Global"],
    "proxy": ["", "", "", "GILTS"],
    "holdings_as_of": ["2026-09-30", "2026-10-29", "", ""],
})

HOLDINGS = pd.DataFrame([
    ["WORLD", "APPLE", "Apple Inc", "AAPL", 0.05, IT, US],
    ["WORLD", "SHELL", "Shell PLC", "SHEL", 0.03, "Energy", UK],
    ["WORLD", "RESIDUAL", "Other (residual)", "", 0.92, "Mixed", "Mixed"],
    ["USTECH", "APPLE", "Apple Inc", "AAPL", 0.40, IT, US],
    ["USTECH", "NVIDIA", "NVIDIA Corp", "NVDA", 0.35, IT, US],
    ["USTECH", "RESIDUAL", "Other (residual)", "", 0.25, "Mixed", "Mixed"],
    ["GILTS", "GILTS", "UK gilts", "GILTS", 1.0, "Bonds", "UK"],
    ["NEWBOND", "NEWBOND", "New bond fund", "NEWBOND", 1.0, "Bonds", "Global"],
], columns=["etf_ticker", "holding_id", "holding_name", "holding_ticker", "weight",
            "sector", "country"])

DAYS = pd.bdate_range("2018-01-01", "2024-12-31")
rng = np.random.default_rng(7)
_market = rng.normal(0.0003, 0.01, len(DAYS))
_crash = (DAYS >= "2020-02-20") & (DAYS <= "2020-03-23")
_market[_crash] = -0.012                                    # a built-in crash
_returns = pd.DataFrame({
    "VWRL.L": _market,
    "WORLD": _market + rng.normal(0, 0.001, len(DAYS)),
    "USTECH": 1.4 * _market + rng.normal(0, 0.004, len(DAYS)),
    "GILTS": rng.normal(0.0001, 0.004, len(DAYS)),
    "NEWBOND": rng.normal(0.0001, 0.003, len(DAYS)),
}, index=DAYS)
PRICES = (1 + _returns).cumprod() * 100
PRICES.loc[PRICES.index < "2021-01-01", "NEWBOND"] = np.nan   # launched 2021

SCENARIOS = pd.DataFrame([
    {"name": "Toy crash", "start_date": "2020-02-19", "end_date": "2020-03-23",
     "reference": "VWRL.L", "kind": "market", "description": ""},
])

WORLD = dict(universe=UNIVERSE, holdings=HOLDINGS, prices=PRICES, scenarios=SCENARIOS)


def portfolio(rows):
    """rows: (ticker, value_gbp)."""
    p = pd.DataFrame(rows, columns=["ticker", "value_gbp"])
    p["weight"] = p["value_gbp"] / p["value_gbp"].sum()
    p["type"], p["holding_id"] = "etf", p["ticker"]
    p["name"] = p["ticker"].map(dict(zip(UNIVERSE["ticker"], UNIVERSE["name"], strict=True)))
    return p


TECH_HEAVY = portfolio([("WORLD", 3000), ("USTECH", 7000)])
BALANCED = portfolio([("WORLD", 5000), ("GILTS", 3000), ("NEWBOND", 2000)])


# --- part 1: offline ------------------------------------------------------------------
def test_report_has_every_part():
    r = run_analysis(TECH_HEAVY, **WORLD)
    assert isinstance(r, Report) and r.ok
    assert r.total_value == 10_000
    assert not r.companies.empty and not r.sectors.empty and not r.countries.empty
    assert list(r.overlap.index) == ["WORLD", "USTECH"]
    assert r.concentration is not None and r.correlation is not None
    assert r.scenario_table["scenario"].tolist() == ["Toy crash"]
    assert r.worst.name == "Toy crash"
    assert len(r.hypotheticals) == 2
    assert r.holdings_as_of == "2026-09-30" and r.prices_as_of == "2024-12-31"


def test_score_breakdown_adds_up():
    r = run_analysis(TECH_HEAVY, **WORLD)
    assert r.score.breakdown["points"].sum() == pytest.approx(r.score.score)
    assert 0 <= r.score.score <= 100


def test_tech_heavy_scores_worse_than_balanced():
    tech = run_analysis(TECH_HEAVY, **WORLD).score.score
    balanced = run_analysis(BALANCED, **WORLD).score.score
    assert tech > balanced


def test_young_fund_proxy_becomes_a_warning():
    r = run_analysis(BALANCED, **WORLD)
    assert any("NEWBOND didn't exist yet in Toy crash" in w for w in r.warnings)


def test_missing_market_prices_skip_one_check_not_the_report():
    r = run_analysis(TECH_HEAVY, **dict(WORLD, prices=PRICES.drop(columns="VWRL.L")))
    assert r.ok
    moves = r.score.breakdown.set_index("component").loc["moves_together"]
    assert not moves["available"]
    assert any("market benchmark" in w for w in r.warnings)


def test_empty_portfolio_is_refused():
    with pytest.raises(ValueError, match="empty"):
        run_analysis(TECH_HEAVY.iloc[0:0], **WORLD)


def test_analyse_csv_carries_errors_and_notices():
    csv = io.StringIO("ticker,value_gbp\nWORLD,6000\nWORLD,1000\nUSTECH,-5\nNOPE,100\n")
    r = analyse_csv(csv, share_lookup=lambda t: None, **WORLD)
    assert r.ok
    assert r.total_value == 7000
    assert len(r.errors) == 2                       # -5 and NOPE
    assert any("appeared 2 times" in n for n in r.notices)


def test_analyse_csv_with_nothing_valid():
    r = analyse_csv(io.StringIO("ticker,value_gbp\nNOPE,100\n"),
                    share_lookup=lambda t: None, **WORLD)
    assert not r.ok and r.errors and r.score is None


def test_share_held_directly_joins_the_fund_company():
    csv = io.StringIO("ticker,value_gbp\nUSTECH,5000\nAAPL,5000\n")
    r = analyse_csv(csv, share_lookup=lambda t: ShareInfo("Apple Inc."), **WORLD)
    apple = r.companies.set_index("holding_id").loc["APPLE"]
    assert apple["exposure"] == pytest.approx(0.5 * 0.40 + 0.5)   # via fund + direct
    assert apple["via"] == "USTECH, direct"


# --- part 2: real data (your machine) -------------------------------------------------
needs_cache = pytest.mark.skipif(not PRICE_CACHE.exists(),
                                 reason="no price cache yet: run scripts/refresh_prices.py")


@needs_cache
@pytest.mark.parametrize("demo", ["naive", "overlapping", "diversified"])
def test_demo_runs_end_to_end_quickly(demo):
    started = time.perf_counter()
    r = analyse_csv(DEMO_DIR / f"{demo}.csv", max_price_age_days=365)
    assert r.ok, r.errors
    assert time.perf_counter() - started < 10, "should finish in under 10 seconds"


@needs_cache
def test_demo_scores_in_the_expected_order():
    scores = {d: analyse_csv(DEMO_DIR / f"{d}.csv", max_price_age_days=365).score.score
              for d in ["overlapping", "naive", "diversified"]}
    assert scores["overlapping"] > scores["naive"] > scores["diversified"], scores