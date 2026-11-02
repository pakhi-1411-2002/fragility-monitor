"""Tests for fragility.lookthrough, built on a toy example worked out on paper.

THE PAPER EXAMPLE
-----------------
Fund A:  Apple 50%, Nvidia 30%, Other (residual) 20%
Fund B:  Apple 40%, Shell  40%, Other (residual) 20%
GOLD:    single block (gold), 100%

Portfolio (£1,000):  A £600 (60%),  B £300 (30%),  Apple shares £100 (10%)

Apple   = 0.6 x 0.5 + 0.3 x 0.4 + 0.1   = 0.30 + 0.12 + 0.10 = 0.52  -> £520
Nvidia  = 0.6 x 0.3                     = 0.18                       -> £180
Shell   = 0.3 x 0.4                     = 0.12                       -> £120
Other   = 0.6 x 0.2 + 0.3 x 0.2         = 0.18                       -> £180
Total   = 0.52 + 0.18 + 0.12 + 0.18     = 1.00

Sectors: tech (Apple + Nvidia) 0.70, energy (Shell) 0.12, unclassified 0.18
Overlap(A, B) = min(0.5, 0.4) [Apple] + min(0.3, 0) + min(0, 0.4) = 0.40
"""

import pandas as pd
import pytest

from fragility.lookthrough import (
    MIXED_COUNTRIES,
    UNCLASSIFIED,
    company_exposure,
    country_exposure,
    explode,
    overlap_matrix,
    sector_exposure,
)

IT, ENERGY, US, UK = "Information Technology", "Energy", "United States", "United Kingdom"

HOLDINGS = pd.DataFrame([
    # etf_ticker, holding_id, holding_name, holding_ticker, weight, sector, country
    ["A", "APPLE", "Apple Inc", "AAPL", 0.5, IT, US],
    ["A", "NVIDIA", "NVIDIA Corp", "NVDA", 0.3, IT, US],
    ["A", "RESIDUAL", "Other (residual)", "", 0.2, "Mixed", "Mixed"],
    ["B", "APPLE", "Apple Inc", "AAPL", 0.4, IT, US],
    ["B", "SHELL", "Shell PLC", "SHEL", 0.4, ENERGY, UK],
    ["B", "RESIDUAL", "Other (residual)", "", 0.2, "Mixed", "Mixed"],
    ["GOLD", "GOLD", "Physical Gold", "GOLD", 1.0, "Commodities", "Global"],
], columns=["etf_ticker", "holding_id", "holding_name", "holding_ticker", "weight",
            "sector", "country"])

UNIVERSE = pd.DataFrame({"ticker": ["A", "B", "GOLD"], "region": ["US", "Global", "Global"]})


def portfolio(rows):
    """rows: (ticker, type, value_gbp, holding_id, name)."""
    df = pd.DataFrame(rows, columns=["ticker", "type", "value_gbp", "holding_id", "name"])
    df["weight"] = df["value_gbp"] / df["value_gbp"].sum()
    return df


PAPER = portfolio([
    ("A", "etf", 600, "A", "Fund A"),
    ("B", "etf", 300, "B", "Fund B"),
    ("AAPL", "share", 100, "APPLE", "Apple Inc."),
])


def as_dict(df, key, value="exposure"):
    return {k: round(v, 6) for k, v in zip(df[key], df[value], strict=True)}


# --- company exposure -------------------------------------------------------
def test_company_exposure_matches_paper():
    out = company_exposure(PAPER, HOLDINGS)
    assert as_dict(out, "holding_id") == {
        "APPLE": 0.52, "NVIDIA": 0.18, "SHELL": 0.12, "RESIDUAL": 0.18}
    assert as_dict(out, "holding_id", "exposure_gbp") == {
        "APPLE": 520, "NVIDIA": 180, "SHELL": 120, "RESIDUAL": 180}


def test_exposures_add_up_to_one():
    assert company_exposure(PAPER, HOLDINGS)["exposure"].sum() == pytest.approx(1.0)


def test_biggest_first_and_residual_last():
    ids = company_exposure(PAPER, HOLDINGS)["holding_id"].tolist()
    assert ids == ["APPLE", "NVIDIA", "SHELL", "RESIDUAL"]


def test_via_lists_every_route_with_direct_last():
    out = company_exposure(PAPER, HOLDINGS).set_index("holding_id")
    assert out.loc["APPLE", "via"] == "A, B, direct"
    assert out.loc["APPLE", "n_routes"] == 3
    assert out.loc["SHELL", "via"] == "B"


def test_direct_share_counts_as_100_percent_itself():
    only_shares = portfolio([("AAPL", "share", 100, "APPLE", "Apple Inc.")])
    out = company_exposure(only_shares, HOLDINGS)
    assert as_dict(out, "holding_id") == {"APPLE": 1.0}
    assert out.loc[0, "sector"] == IT              # sector borrowed from the funds


def test_share_not_in_any_fund_keeps_its_own_name():
    p = portfolio([("TSLA", "share", 100, "TSLA", "Tesla, Inc.")])
    out = company_exposure(p, HOLDINGS)
    assert out.loc[0, ["company", "sector", "kind"]].tolist() == ["Tesla, Inc.", "Unknown",
                                                                  "company"]


def test_residuals_are_never_merged_into_a_company():
    out = company_exposure(PAPER, HOLDINGS).set_index("holding_id")
    assert out.loc["RESIDUAL", "kind"] == "residual"
    assert (out["kind"] == "residual").sum() == 1   # one combined bucket


def test_gold_is_one_block():
    p = portfolio([("A", "etf", 500, "A", "Fund A"), ("GOLD", "etf", 500, "GOLD", "Gold")])
    out = company_exposure(p, HOLDINGS).set_index("holding_id")
    assert out.loc["GOLD", "kind"] == "block" and out.loc["GOLD", "exposure"] == 0.5


def test_fund_without_holdings_data_is_kept_whole():
    p = portfolio([("A", "etf", 500, "A", "Fund A"), ("NEW", "etf", 500, "NEW", "New fund")])
    out = company_exposure(p, HOLDINGS).set_index("holding_id")
    assert out.loc["NEW", "kind"] == "block" and out.loc["NEW", "exposure"] == 0.5
    assert out["exposure"].sum() == pytest.approx(1.0)


# --- sector and country -----------------------------------------------------
def test_sector_exposure_matches_paper():
    assert as_dict(sector_exposure(PAPER, HOLDINGS), "sector") == {
        IT: 0.70, ENERGY: 0.12, UNCLASSIFIED: 0.18}


def test_country_exposure_without_universe():
    assert as_dict(country_exposure(PAPER, HOLDINGS), "country") == {
        US: 0.70, UK: 0.12, MIXED_COUNTRIES: 0.18}


def test_single_country_fund_residual_gets_its_country():
    # Fund A is a US fund, so its 0.12 residual is United States; B's 0.06 stays mixed.
    out = country_exposure(PAPER, HOLDINGS, UNIVERSE)
    assert as_dict(out, "country") == {US: 0.82, UK: 0.12, MIXED_COUNTRIES: 0.06}


# --- overlap ----------------------------------------------------------------
def test_overlap_matches_paper():
    m = overlap_matrix(["A", "B"], HOLDINGS)
    assert m.loc["A", "B"] == pytest.approx(0.40)
    assert m.loc["B", "A"] == pytest.approx(0.40)        # symmetric
    assert m.loc["A", "A"] == m.loc["B", "B"] == 1.0     # a fund fully overlaps itself


def test_no_overlap_between_shares_and_gold():
    assert overlap_matrix(["A", "GOLD"], HOLDINGS).loc["A", "GOLD"] == 0.0


def test_identical_funds_overlap_by_their_named_share():
    twin = HOLDINGS[HOLDINGS["etf_ticker"] == "A"].assign(etf_ticker="A2")
    m = overlap_matrix(["A", "A2"], pd.concat([HOLDINGS, twin]))
    assert m.loc["A", "A2"] == pytest.approx(0.80)       # 0.2 residual can't be compared


# --- explode ----------------------------------------------------------------
def test_explode_has_one_row_per_route():
    lines = explode(PAPER, HOLDINGS)
    assert len(lines) == 3 + 3 + 1                       # A's 3 rows, B's 3, one share
    assert lines["exposure"].sum() == pytest.approx(1.0)


def test_empty_portfolio():
    empty = portfolio([]).astype({"value_gbp": float})
    assert company_exposure(empty, HOLDINGS).empty