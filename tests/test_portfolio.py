"""Tests for fragility.portfolio. Yahoo is replaced by a small fake lookup."""

import io
from pathlib import Path

import pandas as pd
import pytest

from fragility.config import DEMO_DIR, ETF_UNIVERSE_CSV
from fragility.portfolio import (
    TEMPLATE_CSV,
    ShareInfo,
    link_to_holding,
    load_portfolio,
    simple_name,
)

FIXTURES = Path(__file__).parent / "fixtures"

UNIVERSE = pd.DataFrame({
    "ticker": ["VWRL.L", "EQQQ.L", "ISF.L"],
    "name": ["Vanguard FTSE All-World", "Invesco EQQQ Nasdaq-100", "iShares Core FTSE 100"],
})

FAKE_SHARES = {"AAPL": "Apple Inc.", "NVDA": "NVIDIA Corporation",
               "HSBA.L": "HSBC Holdings plc", "TSLA": "Tesla, Inc."}

SYMBOLS = pd.DataFrame({
    "holding_id": ["APPLE", "NVIDIA", "HSBC HOLDINGS", "ARM HOLDINGS"],
    "holding_name": ["Apple Inc", "NVIDIA Corp", "HSBC Holdings PLC", "ARM HOLDINGS PLC ADR"],
    "yahoo_symbol": ["AAPL", "NVDA", "HSBA.L", ""],
})


def fake_lookup(ticker):
    return ShareInfo(FAKE_SHARES[ticker]) if ticker in FAKE_SHARES else None


def load(source, universe=UNIVERSE):
    return load_portfolio(source, universe, share_lookup=fake_lookup, share_symbols=SYMBOLS)


def csv(text):
    return io.StringIO(text)


# --- happy path -------------------------------------------------------------
def test_valid_portfolio_weights_sum_to_one():
    result = load(csv("ticker,value_gbp\nVWRL.L,6000\nEQQQ.L,3000\nAAPL,1000\n"))
    p = result.portfolio
    assert result.errors == [] and result.ok
    assert p["weight"].sum() == pytest.approx(1.0)
    assert dict(zip(p["ticker"], p["weight"], strict=True)) == {
        "VWRL.L": 0.6, "EQQQ.L": 0.3, "AAPL": 0.1}
    assert dict(zip(p["ticker"], p["type"], strict=True)) == {
        "VWRL.L": "etf", "EQQQ.L": "etf", "AAPL": "share"}


def test_template_is_valid():
    result = load(csv(TEMPLATE_CSV))
    assert result.errors == [] and len(result.portfolio) == 3


def test_pound_signs_commas_and_lowercase_are_accepted():
    result = load(csv('ticker,value_gbp\nvwrl.l,"£6,000"\n'))
    assert result.errors == []
    assert result.portfolio.iloc[0][["ticker", "value_gbp"]].tolist() == ["VWRL.L", 6000.0]


def test_missing_dot_l_is_added_with_a_notice():
    result = load(csv("ticker,value_gbp\nVWRL,1000\n"))
    assert result.portfolio["ticker"].tolist() == ["VWRL.L"]
    assert any("VWRL as VWRL.L" in n for n in result.notices)


def test_other_header_names_are_understood():
    result = load(csv("Symbol,Value (£)\nVWRL.L,1000\n"))
    assert result.errors == [] and result.ok


# --- linking shares to fund holdings -----------------------------------------
def test_share_linked_by_ticker():
    assert link_to_holding("AAPL", "Apple Inc.", SYMBOLS) == "APPLE"


def test_share_linked_by_name_when_fund_has_no_ticker():
    assert link_to_holding("ARM", "Arm Holdings plc ADR", SYMBOLS) == "ARM HOLDINGS"


def test_share_not_in_any_fund_keeps_its_own_ticker():
    assert link_to_holding("TSLA", "Tesla, Inc.", SYMBOLS) == "TSLA"


def test_simple_name():
    assert simple_name("Apple Inc.") == simple_name("APPLE INC") == "APPLE"


# --- the five broken files (plan Day 3) ---------------------------------------
def test_blank_rows_are_skipped():
    result = load(FIXTURES / "broken_blank_row.csv")
    assert result.errors == []
    assert result.portfolio["ticker"].tolist() == ["VWRL.L", "EQQQ.L"]


def test_negative_and_zero_values_are_rejected_rest_kept():
    result = load(FIXTURES / "broken_negative_value.csv")
    assert result.portfolio["ticker"].tolist() == ["VWRL.L"]
    assert len(result.errors) == 2
    assert all("more than £0" in e for e in result.errors)
    assert result.errors[0].startswith("Row 3 (EQQQ.L)")


def test_unknown_ticker_is_rejected_with_a_clear_message():
    result = load(FIXTURES / "broken_unknown_ticker.csv")
    assert result.portfolio["ticker"].tolist() == ["VWRL.L"]
    assert result.errors == [result.errors[0]]
    assert "NOTREAL1" in result.errors[0] and "not in our fund list" in result.errors[0]


def test_duplicates_are_merged_with_a_notice():
    result = load(FIXTURES / "broken_duplicates.csv")
    p = result.portfolio.set_index("ticker")
    assert p.loc["VWRL.L", "value_gbp"] == 10000
    assert any("VWRL.L appeared 2 times" in n for n in result.notices)
    assert p["weight"].sum() == pytest.approx(1.0)


def test_wrong_headers_give_one_clear_error():
    result = load(FIXTURES / "broken_wrong_headers.csv")
    assert not result.ok
    assert len(result.errors) == 1 and "'ticker' and 'value_gbp'" in result.errors[0]


def test_empty_file():
    result = load(csv(""))
    assert not result.ok and result.errors == ["The file is empty."]


# --- demo portfolios against the real universe --------------------------------
@pytest.mark.skipif(not ETF_UNIVERSE_CSV.exists(), reason="universe not built yet")
@pytest.mark.parametrize("demo", ["naive", "overlapping", "diversified"])
def test_demo_portfolios_load_cleanly(demo):
    universe = pd.read_csv(ETF_UNIVERSE_CSV, dtype=str).fillna("")
    result = load(DEMO_DIR / f"{demo}.csv", universe)
    assert result.errors == [], result.errors
    assert result.portfolio["value_gbp"].sum() == 25_000
    assert result.portfolio["weight"].sum() == pytest.approx(1.0)
