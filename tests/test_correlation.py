"""Tests for fragility.correlation. Synthetic data with known answers, no internet."""

import numpy as np
import pandas as pd
import pytest

from fragility.correlation import (
    cluster_order,
    clusters,
    corr_matrix,
    correlation_summary,
    diversification_ratio,
    effective_groups,
    short_history,
    stress_corr_matrix,
    weighted_avg_corr,
)

WEEKS = pd.date_range("2020-01-03", periods=260, freq="W-FRI")   # 5 years of Fridays
rng = np.random.default_rng(42)


def noise(scale=0.02, n=None):
    return rng.normal(0, scale, len(WEEKS) if n is None else n)


# --- correlation matrices ----------------------------------------------------
def test_identical_series_have_correlation_one():
    x = noise()
    r = pd.DataFrame({"A": x, "B": x}, index=WEEKS)
    assert corr_matrix(r).loc["A", "B"] == pytest.approx(1.0)


def test_mirrored_series_have_correlation_minus_one():
    x = noise()
    r = pd.DataFrame({"A": x, "B": -x}, index=WEEKS)
    assert corr_matrix(r).loc["A", "B"] == pytest.approx(-1.0)


def test_window_uses_only_the_last_years():
    # First 2 years mirror each other, last 3 years identical -> 3-year window says +1.
    x = noise()
    b = np.where(WEEKS < WEEKS[-1] - pd.DateOffset(years=3), -x, x)
    r = pd.DataFrame({"A": x, "B": b}, index=WEEKS)
    assert corr_matrix(r, years=3).loc["A", "B"] == pytest.approx(1.0)
    assert corr_matrix(r, years=None).loc["A", "B"] < 0.5


def test_too_little_shared_history_gives_nan():
    r = pd.DataFrame({"A": noise(), "B": [np.nan] * 250 + list(noise(n=10))}, index=WEEKS)
    assert np.isnan(corr_matrix(r).loc["A", "B"])


def test_stress_correlation_only_looks_at_bad_weeks():
    # Gold (G) is unrelated in normal weeks but drops WITH the market in bad weeks.
    market = noise(0.025)
    bad = market < -0.02
    gold = np.where(bad, market * 0.8, noise(0.01))
    r = pd.DataFrame({"MKT": market, "G": gold}, index=WEEKS)
    assert stress_corr_matrix(r, market="MKT").loc["MKT", "G"] == pytest.approx(1.0)
    assert corr_matrix(r, years=None).loc["MKT", "G"] < 0.9


def test_stress_needs_the_market_column():
    with pytest.raises(KeyError, match="VWRL.L"):
        stress_corr_matrix(pd.DataFrame({"A": noise()}, index=WEEKS))


# --- weighted average correlation --------------------------------------------
CORR = pd.DataFrame([[1.0, 0.8, 0.2],
                     [0.8, 1.0, 0.0],
                     [0.2, 0.0, 1.0]], index=list("ABC"), columns=list("ABC"))


def test_weighted_average_by_hand():
    # Pairs (counted both ways): AB w=.5*.3=.15 rho .8 | AC .5*.2=.10 rho .2 | BC .3*.2=.06 rho 0
    #   (.15*.8 + .10*.2 + .06*0) / (.15 + .10 + .06) = .14 / .31 = 0.4516
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    assert weighted_avg_corr(CORR, w) == pytest.approx(0.14 / 0.31)


def test_weighted_average_skips_unknown_pairs():
    c = CORR.copy()
    c.loc["A", "C"] = c.loc["C", "A"] = np.nan
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    assert weighted_avg_corr(c, w) == pytest.approx(0.12 / 0.21)


def test_one_holding_has_no_average():
    assert np.isnan(weighted_avg_corr(CORR, pd.Series({"A": 1.0})))


# --- diversification ratio ---------------------------------------------------
def test_identical_holdings_give_no_diversification():
    x = noise()
    r = pd.DataFrame({"A": x, "B": x}, index=WEEKS)
    assert diversification_ratio(r, pd.Series({"A": 0.5, "B": 0.5})) == pytest.approx(1.0)


def test_unrelated_equal_holdings_give_about_root_two():
    r = pd.DataFrame({"A": noise(), "B": noise()}, index=WEEKS)
    ratio = diversification_ratio(r, pd.Series({"A": 0.5, "B": 0.5}), years=None)
    assert ratio == pytest.approx(np.sqrt(2), abs=0.1)


# --- clusters ------------------------------------------------------------------
def test_clusters_group_close_holdings():
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    c = CORR.copy()
    c.loc["A", "B"] = c.loc["B", "A"] = 0.95
    assert clusters(c, weights=w) == [["A", "B"], ["C"]]


def test_loosely_related_holdings_stay_apart():
    assert len(clusters(CORR)) == 3        # rho 0.8 is just below the ~0.82 cut


def test_single_holding_is_one_group():
    assert clusters(CORR.loc[["A"], ["A"]]) == [["A"]]


def test_effective_groups_by_hand():
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    # groups weigh 0.8 and 0.2 -> 1 / (0.64 + 0.04) = 1.47
    assert effective_groups([["A", "B"], ["C"]], w) == pytest.approx(1 / 0.68)


def test_cluster_order_puts_similar_holdings_side_by_side():
    c = pd.DataFrame(np.eye(4), index=list("WXYZ"), columns=list("WXYZ"))
    c.loc["W", "Y"] = c.loc["Y", "W"] = 0.95
    order = cluster_order(c)
    assert abs(order.index("W") - order.index("Y")) == 1


# --- history ---------------------------------------------------------------------
def test_short_history_flag():
    days = pd.bdate_range("2020-01-01", "2024-12-31")
    p = pd.DataFrame({"OLD": 1.0, "NEW": np.nan}, index=days)
    p.loc[p.index >= "2024-01-01", "NEW"] = 1.0
    assert short_history(p, min_years=2) == ["NEW"]


# --- everything together -----------------------------------------------------------
def test_summary_on_synthetic_portfolio():
    days = pd.bdate_range("2019-01-01", "2024-12-31")
    market = rng.normal(0.0003, 0.01, len(days))
    us1 = market + rng.normal(0, 0.001, len(days))     # S&P-like fund
    us2 = market + rng.normal(0, 0.001, len(days))     # another S&P-like fund
    gold = rng.normal(0.0002, 0.008, len(days))        # unrelated
    prices = pd.DataFrame({"VWRL.L": market, "US1": us1, "US2": us2, "GOLD": gold},
                          index=days).add(1).cumprod() * 100

    weights = pd.Series({"US1": 0.4, "US2": 0.4, "GOLD": 0.2, "NOPRICE": 0.1})
    s = correlation_summary(prices, weights)

    assert s.missing_prices == ["NOPRICE"]
    assert s.groups == [["US1", "US2"], ["GOLD"]]
    assert s.effective_groups == pytest.approx(1 / (0.8 ** 2 + 0.2 ** 2), rel=0.01)
    assert 0.5 < s.avg_corr < 0.8            # US1-US2 ~0.96, both vs gold ~0
    assert 1.0 < s.diversification_ratio < 1.5
    assert s.n_stress_weeks > 0