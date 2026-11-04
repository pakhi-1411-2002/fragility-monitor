"""Tests for fragility.prices. No internet needed: Yahoo is replaced by fake data."""

import numpy as np
import pandas as pd
import pytest

from fragility import prices

DAYS = pd.bdate_range("2024-01-01", periods=10)


def series(values, name="X.L"):
    return pd.Series(values, index=DAYS[: len(values)], name=name, dtype=float)


# --- pence glitches ---------------------------------------------------------
def test_pence_glitch_in_the_middle_is_repaired():
    s = series([138, 139, 13900, 14000, 140, 141])     # two days in pence by mistake
    fixed = prices.fix_pence_glitches(s)
    assert list(fixed.round(2)) == [138, 139, 139, 140, 140, 141]


def test_glitch_at_the_start_uses_the_scale_of_most_days():
    s = series([13800, 139, 140, 141, 142])            # first day in pence
    assert list(prices.fix_pence_glitches(s).round(2)) == [138, 139, 140, 141, 142]


def test_real_big_moves_are_left_alone():
    s = series([100, 60, 30, 45, 90])                  # -40%, -50%, +50%, +100%
    pd.testing.assert_series_equal(prices.fix_pence_glitches(s), s)


# --- currency conversion ----------------------------------------------------
def test_pence_and_dollars_become_pounds():
    raw = pd.DataFrame({"P.L": [13800.0, 14000.0],   # pence
                        "D.L": [500.0, 510.0],       # dollars
                        "G.L": [50.0, 51.0]},        # pounds
                       index=DAYS[:2])
    fx = pd.DataFrame({"GBPUSD=X": [1.25, 1.275]}, index=DAYS[:2])
    gbp = prices.to_gbp(raw, {"P.L": "GBp", "D.L": "USD", "G.L": "GBP"}, fx)
    assert list(gbp["P.L"]) == [138.0, 140.0]
    assert list(gbp["D.L"]) == [400.0, 400.0]        # 500/1.25, 510/1.275
    assert list(gbp["G.L"]) == [50.0, 51.0]


def test_missing_exchange_rate_is_an_error():
    raw = pd.DataFrame({"E.DE": [10.0]}, index=DAYS[:1])
    with pytest.raises(KeyError, match="GBPEUR=X"):
        prices.to_gbp(raw, {"E.DE": "EUR"}, pd.DataFrame(index=DAYS[:1]))


def test_guess_currency():
    assert prices.guess_currency("VWRL.L") == "GBp"
    assert prices.guess_currency("AAPL") == "USD"


# --- cleaning ---------------------------------------------------------------
TRADING = list(range(1, 11))   # a ticker that trades every day, so no date row is empty


def test_short_gaps_are_filled_long_gaps_are_not():
    p = pd.DataFrame({"A": [1, np.nan, np.nan, 4, 5, 6, 7, 8, 9, 10], "T": TRADING},
                     index=DAYS, dtype=float)
    clean, dropped = prices.clean_prices(p)
    assert clean["A"].isna().sum() == 0 and dropped == []

    gappy = pd.DataFrame({"B": [1] + [np.nan] * 8 + [10], "T": TRADING},
                         index=DAYS, dtype=float)
    clean, dropped = prices.clean_prices(gappy)       # 8-day gap > 5-day fill limit
    assert dropped == ["B"]


def test_new_fund_is_not_punished_for_days_before_it_existed():
    p = pd.DataFrame({"OLD": range(1, 11), "NEW": [np.nan] * 6 + [1, 2, 3, 4]},
                     index=DAYS, dtype=float)
    clean, dropped = prices.clean_prices(p)
    assert dropped == [] and "NEW" in clean.columns


def test_zero_prices_are_treated_as_missing():
    p = pd.DataFrame({"A": [10, 0, 11, 12, 13, 14, 15, 16, 17, 18], "T": TRADING},
                     index=DAYS, dtype=float)
    clean, _ = prices.clean_prices(p)
    assert clean["A"].iloc[1] == 10                     # filled from the day before


# --- returns ----------------------------------------------------------------
def test_daily_returns():
    p = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=DAYS[:3])
    assert list(prices.returns(p)["A"].round(4)) == [0.10, -0.10]


def test_weekly_returns_use_friday_closes():
    days = pd.bdate_range("2024-01-01", periods=10)     # Mon 1 Jan .. Fri 12 Jan
    p = pd.DataFrame({"A": [100.0] * 5 + [100, 100, 100, 100, 120]}, index=days)
    assert prices.weekly_returns(p)["A"].round(4).tolist() == [0.20]


# --- download + cache (fake Yahoo) -------------------------------------------
def fake_yahoo(calls):
    def download(tickers, start):
        calls.append(sorted(tickers))
        data = {t: [13800.0, 13900.0] if t.endswith(".L") else [1.25, 1.25]
                for t in tickers}
        frame = pd.DataFrame(data, index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"]))
        frame.columns = pd.MultiIndex.from_product([["Close"], frame.columns])
        return frame
    return download


@pytest.fixture
def offline(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(prices, "PRICE_CACHE", tmp_path / "prices.parquet")
    monkeypatch.setattr(prices, "_yf_download", fake_yahoo(calls))
    monkeypatch.setattr(prices, "get_currencies", lambda ts: {t: "GBp" for t in ts})
    return calls


def test_first_call_downloads_and_caches(offline):
    p = prices.load_prices(["AAA.L"], start="2024-01-01")
    assert p["AAA.L"].tolist() == [138.0, 139.0]
    assert prices.PRICE_CACHE.exists() and len(offline) == 1


def test_second_call_reads_from_cache(offline):
    prices.load_prices(["AAA.L"], start="2024-01-01")
    prices.load_prices(["AAA.L"], start="2024-01-01")
    assert len(offline) == 1                            # no second download


def test_new_ticker_downloads_only_that_ticker(offline):
    prices.load_prices(["AAA.L"], start="2024-01-01")
    p = prices.load_prices(["AAA.L", "BBB.L"], start="2024-01-01")
    assert offline[-1] == ["BBB.L"] and set(p.columns) == {"AAA.L", "BBB.L"}


def test_refresh_downloads_everything_again(offline):
    prices.load_prices(["AAA.L"], start="2024-01-01")
    prices.load_prices(["AAA.L"], start="2024-01-01", refresh=True)
    assert len(offline) == 2


# --- one-day spikes ------------------------------------------------------------
def test_one_day_spike_that_reverses_is_removed():
    # The real HMWO.L case from 24 Oct 2025: 32.70 -> 43.92 -> 33.23
    s = series([32.53, 32.70, 43.92, 33.23, 33.50])
    cleaned, removed = prices.remove_spikes(s)
    assert removed == [DAYS[2]] and np.isnan(cleaned.iloc[2])
    assert cleaned.drop(DAYS[2]).equals(s.drop(DAYS[2]))


def test_spike_down_is_removed_too():
    cleaned, removed = prices.remove_spikes(series([100, 101, 60, 100, 102]))
    assert removed == [DAYS[2]]


def test_real_crash_is_kept():
    # A big fall that STAYS down is a real move, not a glitch.
    s = series([100, 100, 75, 74, 76])
    assert prices.remove_spikes(s)[1] == []


def test_spike_is_filled_from_the_day_before_after_cleaning():
    p = pd.DataFrame({"A": [32.53, 32.70, 43.92, 33.23, 33.50, 33.6, 33.7, 33.8, 33.9, 34.0],
                      "T": TRADING}, index=DAYS, dtype=float)
    clean, _ = prices.clean_prices(p)
    assert clean["A"].iloc[2] == 32.70