"""Tests for fragility.concentration, with answers worked out by hand."""

import pandas as pd
import pytest

from fragility.concentration import (
    apparent_vs_real,
    concentration_summary,
    effective_n,
    hhi,
    max_country_share,
    max_sector_share,
    top_n_share,
)
from fragility.lookthrough import MIXED_COUNTRIES, UNCLASSIFIED

IT, US, UK = "Information Technology", "United States", "United Kingdom"


def exposures(rows):
    """rows: (exposure, kind) -> a table shaped like company_exposure's output."""
    return pd.DataFrame(rows, columns=["exposure", "kind"])


# --- HHI and effective number of bets ---------------------------------------
def test_ten_equal_companies_are_ten_bets():
    e = exposures([(0.1, "company")] * 10)
    assert hhi(e) == pytest.approx(0.1)
    assert effective_n(e) == pytest.approx(10)


def test_one_company_is_one_bet():
    e = exposures([(1.0, "company")])
    assert hhi(e) == 1.0 and effective_n(e) == 1.0


def test_uneven_shares():
    # 0.5^2 + 0.3^2 + 0.2^2 = 0.25 + 0.09 + 0.04 = 0.38  ->  1 / 0.38 = 2.63 bets
    e = exposures([(0.5, "company"), (0.3, "company"), (0.2, "company")])
    assert hhi(e) == pytest.approx(0.38)
    assert effective_n(e) == pytest.approx(1 / 0.38)


def test_residual_is_ignored_not_renormalised():
    # One company at 50%, the other 50% spread over thousands of tiny companies.
    # HHI = 0.5^2 = 0.25 (the tiny ones add ~0), so N_eff = 4, not 1.
    e = exposures([(0.5, "company"), (0.5, "residual")])
    assert hhi(e) == pytest.approx(0.25)
    assert effective_n(e) == pytest.approx(4)


def test_bond_or_gold_block_counts_as_one_bet():
    e = exposures([(0.5, "company"), (0.5, "block")])
    assert effective_n(e) == pytest.approx(2)


def test_empty_portfolio_has_zero_bets():
    assert effective_n(exposures([])) == 0.0


# --- top N --------------------------------------------------------------------
def test_top_n_share():
    e = exposures([(0.3, "company"), (0.2, "company"), (0.1, "company"), (0.4, "residual")])
    assert top_n_share(e, n=2) == pytest.approx(0.5)


def test_top_n_with_fewer_than_n_companies_takes_them_all():
    e = exposures([(0.3, "company"), (0.2, "company"), (0.5, "residual")])
    assert top_n_share(e, n=10) == pytest.approx(0.5)


def test_top_n_ignores_bond_and_gold_blocks():
    e = exposures([(0.6, "block"), (0.4, "company")])
    assert top_n_share(e, n=10) == pytest.approx(0.4)


# --- sector and country tilt -------------------------------------------------
def test_max_sector_skips_unclassified_bonds_and_unknown():
    sectors = pd.DataFrame({
        "sector": [UNCLASSIFIED, "Bonds", IT, "Energy", "Unknown"],
        "exposure": [0.40, 0.25, 0.20, 0.10, 0.05],
    })
    assert max_sector_share(sectors) == (IT, pytest.approx(0.20))


def test_max_country_merges_uk_alias_and_skips_mixed():
    countries = pd.DataFrame({
        "country": [US, MIXED_COUNTRIES, UK, "UK", "Global"],
        "exposure": [0.30, 0.25, 0.20, 0.15, 0.10],
    })
    # UK fund block ("UK") + UK companies -> 0.35, beats the US at 0.30
    assert max_country_share(countries) == (UK, pytest.approx(0.35))


def test_no_real_sector_returns_blank():
    assert max_sector_share(pd.DataFrame({"sector": ["Bonds"], "exposure": [1.0]})) == ("", 0.0)


# --- apparent vs real (uses look-through) ------------------------------------
HOLDINGS = pd.DataFrame([
    ["A", "APPLE", "Apple Inc", 0.5, IT, US],
    ["A", "NVIDIA", "NVIDIA Corp", 0.3, IT, US],
    ["A", "RESIDUAL", "Other (residual)", 0.2, "Mixed", "Mixed"],
    ["B", "APPLE", "Apple Inc", 0.4, IT, US],
    ["B", "SHELL", "Shell PLC", 0.4, "Energy", UK],
    ["B", "RESIDUAL", "Other (residual)", 0.2, "Mixed", "Mixed"],
], columns=["etf_ticker", "holding_id", "holding_name", "weight", "sector", "country"])

HALF_AND_HALF = pd.DataFrame({
    "ticker": ["A", "B"], "type": ["etf", "etf"], "value_gbp": [500.0, 500.0],
    "weight": [0.5, 0.5], "holding_id": ["A", "B"], "name": ["Fund A", "Fund B"],
})


def test_apparent_vs_real_by_hand():
    # Apparent: Apple-in-A 0.25, Nvidia 0.15, Apple-in-B 0.20, Shell 0.20
    #   HHI = 0.0625 + 0.0225 + 0.04 + 0.04 = 0.165  -> 6.06 bets
    # Real:   Apple 0.45, Nvidia 0.15, Shell 0.20
    #   HHI = 0.2025 + 0.0225 + 0.04 = 0.265         -> 3.77 bets
    out = apparent_vs_real(HALF_AND_HALF, HOLDINGS)
    assert out["n_lines"] == 2
    assert out["n_eff_apparent"] == pytest.approx(1 / 0.165)
    assert out["n_eff_real"] == pytest.approx(1 / 0.265)
    assert out["overlap_loss"] == pytest.approx(1 - 0.165 / 0.265)


def test_no_overlap_means_no_loss():
    only_a = HALF_AND_HALF.iloc[[0]].assign(weight=1.0, value_gbp=1000.0)
    assert apparent_vs_real(only_a, HOLDINGS)["overlap_loss"] == pytest.approx(0)


def test_summary_runs_end_to_end():
    s = concentration_summary(HALF_AND_HALF, HOLDINGS)
    assert s.effective_n == pytest.approx(1 / 0.265)
    assert s.top10_share == pytest.approx(0.80)        # Apple + Nvidia + Shell
    assert (s.top_sector, round(s.top_sector_share, 2)) == (IT, 0.60)
    assert set(s.to_dict()) >= {"hhi", "effective_n", "top10_share", "overlap_loss"}