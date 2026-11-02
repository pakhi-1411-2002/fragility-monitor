"""Tests for fragility.stress, with hand-made prices and answers worked out by hand."""

import numpy as np
import pandas as pd
import pytest

from fragility.stress import (
    hypothetical_shock,
    max_drawdown_in_window,
    replay,
    replay_all,
    run_hypotheticals,
    scenario_table,
    tickers_needed,
    worst_scenario,
)

DAYS = pd.bdate_range("2020-02-17", periods=6)        # Mon 17 Feb .. Mon 24 Feb 2020
SCENARIO = pd.Series({"name": "Toy crash", "kind": "market",
                      "start_date": "2020-02-17", "end_date": "2020-02-24"})

# A falls 100 -> 80 (-20%), dipping to 70 on the way. B rises 100 -> 110 (+10%).
PRICES = pd.DataFrame({
    "A": [100, 90, 70, 75, 78, 80],
    "B": [100, 102, 104, 106, 108, 110],
    "FLAT": [50] * 6,
}, index=DAYS, dtype=float)


def portfolio(rows):
    """rows: (ticker, value_gbp)."""
    df = pd.DataFrame(rows, columns=["ticker", "value_gbp"])
    df["weight"] = df["value_gbp"] / df["value_gbp"].sum()
    return df


HALF_HALF = portfolio([("A", 500), ("B", 500)])


# --- peak to trough -------------------------------------------------------------
def test_max_drawdown_simple():
    v = pd.Series([100, 120, 90, 110, 60, 80], index=pd.bdate_range("2020-01-01", periods=6))
    out = max_drawdown_in_window(v)
    assert out["fall"] == pytest.approx(60 / 120 - 1)       # -50%: from 120 down to 60
    assert out["peak_date"] == v.index[1] and out["trough_date"] == v.index[4]


def test_max_drawdown_rising_market_is_zero():
    v = pd.Series([1.0, 2.0, 3.0], index=pd.bdate_range("2020-01-01", periods=3))
    assert max_drawdown_in_window(v)["fall"] == 0.0


# --- replay -------------------------------------------------------------------------
def test_replay_by_hand():
    # 50% in A (-20%) + 50% in B (+10%) = -10% + 5% = -5%  ->  -£50 on £1,000
    r = replay(HALF_HALF, PRICES, SCENARIO)
    assert r.portfolio_return == pytest.approx(-0.05)
    assert r.loss_gbp == pytest.approx(-50)
    assert r.coverage == 1.0 and r.estimated == [] and r.excluded == []


def test_worst_dip_inside_the_window():
    # Day by day: 0.5*A/100 + 0.5*B/100 = 1.00, 0.96, 0.87, 0.905, 0.93, 0.95
    # The worst dip is 1.00 -> 0.87 = -13%, deeper than the -5% at the end.
    r = replay(HALF_HALF, PRICES, SCENARIO)
    assert r.max_drawdown == pytest.approx(-0.13)
    assert r.worst_loss == pytest.approx(-0.13)
    assert r.path.round(3).tolist() == [1.0, 0.96, 0.87, 0.905, 0.93, 0.95]


def test_contributors_worst_first():
    r = replay(HALF_HALF, PRICES, SCENARIO)
    top = r.contributors.iloc[0]
    assert top["ticker"] == "A"
    assert top["contribution"] == pytest.approx(-0.10)
    assert top["loss_gbp"] == pytest.approx(-100)


def test_young_fund_borrows_its_proxy_and_is_flagged():
    prices = PRICES.assign(YOUNG=np.nan)
    prices.loc[DAYS[3]:, "YOUNG"] = 10.0                       # launched mid-crash
    universe = pd.DataFrame({"ticker": ["YOUNG"], "proxy": ["A"]})
    r = replay(portfolio([("YOUNG", 100)]), prices, SCENARIO, universe)
    assert r.estimated == ["YOUNG"]
    assert r.portfolio_return == pytest.approx(-0.20)          # A's move


def test_holding_without_prices_or_proxy_is_left_out():
    # A has no prices and no proxy: B carries the result, coverage shows 50%.
    p = portfolio([("NOPRICE", 500), ("B", 500)])
    r = replay(p, PRICES, SCENARIO)
    assert r.excluded == ["NOPRICE"]
    assert r.coverage == pytest.approx(0.5)
    assert r.portfolio_return == pytest.approx(0.10)


def test_nothing_priced_gives_no_result():
    r = replay(portfolio([("NOPE", 100)]), PRICES, SCENARIO)
    assert np.isnan(r.portfolio_return) and r.coverage == 0.0


def test_start_on_a_weekend_uses_fridays_close():
    weekend = SCENARIO.copy()
    weekend["start_date"] = "2020-02-16"                       # a Sunday: no price
    prices = pd.concat([pd.DataFrame({"A": [100.0], "B": [100.0]},
                                     index=[pd.Timestamp("2020-02-14")]), PRICES])
    r = replay(HALF_HALF, prices, weekend)
    assert r.portfolio_return == pytest.approx(-0.05)


def test_flat_prices_lose_nothing():
    r = replay(portfolio([("FLAT", 100)]), PRICES, SCENARIO)
    assert r.portfolio_return == 0.0 and r.max_drawdown == 0.0


# --- several scenarios -----------------------------------------------------------------
def test_replay_all_table_and_worst():
    calm = SCENARIO.copy()
    calm["name"], calm["end_date"] = "Calm", "2020-02-18"     # A -10%, B +2% -> -4%
    scenarios = pd.DataFrame([SCENARIO, calm])
    results = replay_all(HALF_HALF, PRICES, scenarios)
    table = scenario_table(results)
    assert table["scenario"].tolist() == ["Toy crash", "Calm"]
    assert table.loc[0, "worst_contributors"] == "A"
    assert worst_scenario(results).name == "Toy crash"


def test_tickers_needed_includes_proxies():
    universe = pd.DataFrame({"ticker": ["VUAG.L", "VWRL.L"], "proxy": ["VUSA.L", ""]})
    p = portfolio([("VUAG.L", 1), ("VWRL.L", 1)])
    assert tickers_needed(p, universe) == ["VUAG.L", "VWRL.L", "VUSA.L"]


# --- hypothetical shocks ---------------------------------------------------------------
IT, US, UK = "Information Technology", "United States", "United Kingdom"
LINES = pd.DataFrame([
    # sector, country, kind, exposure
    [IT, US, "company", 0.40],
    [IT, UK, "company", 0.10],
    ["Energy", UK, "company", 0.20],
    ["Bonds", UK, "block", 0.30],               # a UK gilt fund: not an equity
], columns=["sector", "country", "kind", "exposure"])


def test_us_tech_shock_by_hand():
    # Only US tech is hit: 0.40 x -20% = -8%
    out = hypothetical_shock(LINES, -0.20, sector=IT, country=US)
    assert out == {"exposed": pytest.approx(0.40), "loss": pytest.approx(-0.08)}


def test_uk_equity_shock_skips_the_gilt_fund():
    # UK companies 0.10 + 0.20 = 0.30 (gilts excluded) x -15% = -4.5%
    out = hypothetical_shock(LINES, -0.15, country=UK)
    assert out["exposed"] == pytest.approx(0.30) and out["loss"] == pytest.approx(-0.045)


def test_run_hypotheticals_in_pounds():
    holdings = pd.DataFrame([
        ["F", "APPLE", "Apple Inc", 0.6, IT, US],
        ["F", "SHELL", "Shell PLC", 0.4, "Energy", UK],
    ], columns=["etf_ticker", "holding_id", "holding_name", "weight", "sector", "country"])
    p = pd.DataFrame({"ticker": ["F"], "type": ["etf"], "value_gbp": [10_000.0],
                      "weight": [1.0], "holding_id": ["F"], "name": ["Fund F"]})
    out = run_hypotheticals(p, holdings).set_index("scenario")
    assert out.loc["US tech -20%", "loss_gbp"] == pytest.approx(-1_200)   # 60% x -20%
    assert out.loc["UK equities -15%", "loss_gbp"] == pytest.approx(-600)  # 40% x -15%