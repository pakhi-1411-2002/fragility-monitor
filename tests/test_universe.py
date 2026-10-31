"""Tests for the reference-data checks in fragility.universe."""

import pandas as pd
import pytest

from fragility.config import ETF_HOLDINGS_CSV
from fragility.universe import (
    load_holdings,
    load_universe,
    missing_tickers,
    validate_holdings,
)


def toy_universe():
    return pd.DataFrame({
        "ticker": ["AAA.L", "GLD.L"],
        "asset_class": ["equity", "commodity"],
    })


def toy_holdings(changes=None):
    """Small valid table; `changes` = {(row, column): new value} to break it."""
    rows = [
        # etf_ticker, holding_id, holding_name, holding_ticker, weight
        ["AAA.L", "APPLE", "Apple Inc", "AAPL", 0.60],
        ["AAA.L", "NVIDIA", "NVIDIA Corp", "NVDA", 0.30],
        ["AAA.L", "RESIDUAL", "Other (residual)", "", 0.10],
        ["GLD.L", "GLD.L", "Gold ETF", "GLD.L", 1.00],
    ]
    df = pd.DataFrame(rows, columns=["etf_ticker", "holding_id", "holding_name",
                                     "holding_ticker", "weight"])
    for (row, col), value in (changes or {}).items():
        df.loc[row, col] = value
    return df


def test_clean_data_has_no_problems():
    assert validate_holdings(toy_holdings(), toy_universe()) == []


def test_weights_not_summing_to_one_are_caught():
    bad = toy_holdings({(0, "weight"): 0.50})  # AAA.L now sums to 0.90
    problems = validate_holdings(bad, toy_universe())
    assert any("AAA.L: weights add up to 0.9000" in p for p in problems)


def test_small_rounding_is_allowed():
    ok = toy_holdings({(0, "weight"): 0.603})  # sums to 1.003, inside +/-0.005
    assert validate_holdings(ok, toy_universe()) == []


def test_blank_ticker_is_a_warning_not_a_failure():
    data = toy_holdings({(1, "holding_ticker"): ""})
    assert validate_holdings(data, toy_universe()) == []
    assert list(missing_tickers(data)["holding_id"]) == ["NVIDIA"]  # residual not listed


def test_missing_etf_is_caught():
    holdings = toy_holdings().query("etf_ticker != 'GLD.L'")
    assert "GLD.L: no holdings at all" in validate_holdings(holdings, toy_universe())


def test_duplicate_company_is_caught():
    bad = toy_holdings({(1, "holding_id"): "APPLE"})
    problems = validate_holdings(bad, toy_universe())
    assert any("APPLE appears more than once" in p for p in problems)


@pytest.mark.skipif(not ETF_HOLDINGS_CSV.exists(), reason="holdings not built yet")
def test_real_reference_data_passes():
    problems = validate_holdings(load_holdings(), load_universe())
    assert problems == [], "\n".join(problems)