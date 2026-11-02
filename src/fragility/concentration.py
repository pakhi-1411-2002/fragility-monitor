"""Concentration: how many genuinely separate bets is the money spread over?

Plain idea: twelve apples, a carton of apple juice and an apple pie are still
mostly one fruit. If apples go bad, everything goes bad together. These
functions count how many truly separate "fruits" the money is spread over.

The key number is the HHI (Herfindahl-Hirschman Index): square every share
and add them up.

    HHI   = sum over companies c of  E_c ^ 2
    N_eff = 1 / HHI        ("effective number of bets")

Ten equal 10% holdings:  HHI = 10 x 0.1^2 = 0.1   -> N_eff = 10
One 100% holding:        HHI = 1^2        = 1.0   -> N_eff = 1
Squaring makes big holdings count much more than small ones, so N_eff drops
fast when a few names dominate.

Inputs are the tables made by lookthrough.py. No file reading, no Streamlit.
"""

from dataclasses import asdict, dataclass

import pandas as pd

from fragility.lookthrough import (
    BLOCK,
    COMPANY,
    MIXED_COUNTRIES,
    UNCLASSIFIED,
    company_exposure,
    country_exposure,
    explode,
    sector_exposure,
)

TOP_N = 10

# Rows that are not a real sector or country, so never count as a "tilt".
NOT_A_SECTOR = {UNCLASSIFIED, "Unknown", "Bonds", "Commodities", "Other", "Mixed", ""}
NOT_A_COUNTRY = {MIXED_COUNTRIES, "Unknown", "Mixed", "Global", "Developed", "Emerging",
                 "Europe", ""}
# Bond and gold funds carry their fund region ("UK"), not a country name.
COUNTRY_ALIASES = {"UK": "United Kingdom", "US": "United States"}


# ---------------------------------------------------------------------------
# The basic maths
# ---------------------------------------------------------------------------
def bets(exposures: pd.DataFrame) -> pd.Series:
    """The separate bets: companies plus whole blocks (a bond fund, gold).

    The residual bucket is left out on purpose, and NOT renormalised: it is
    thousands of tiny companies, each far too small to add anything to the
    HHI (3,800 holdings of 0.015% add about 0.0001). Renormalising would
    instead blow the top 50 up as if they were the whole fund.
    """
    keep = exposures["kind"].isin([COMPANY, BLOCK])
    return exposures.loc[keep, "exposure"].astype(float)


def hhi(exposures: pd.DataFrame) -> float:
    """Sum of squared shares. 0 = spread over countless tiny bets, 1 = one bet."""
    return float((bets(exposures) ** 2).sum())


def effective_n(exposures: pd.DataFrame) -> float:
    """1 / HHI: how many equal-sized, separate bets the money behaves like."""
    h = hhi(exposures)
    return 1 / h if h > 0 else 0.0


def top_n_share(exposures: pd.DataFrame, n: int = TOP_N) -> float:
    """Share of the money in the n biggest companies (bond/gold blocks excluded)."""
    companies = exposures.loc[exposures["kind"] == COMPANY, "exposure"].astype(float)
    return float(companies.nlargest(n).sum())


def _max_share(table: pd.DataFrame, column: str, exclude: set,
               aliases: dict | None = None) -> tuple:
    """(biggest label, its share) after dropping rows that aren't real labels."""
    labels = table[column].astype(str).map(lambda v: (aliases or {}).get(v, v))
    grouped = table.assign(**{column: labels}).groupby(column)["exposure"].sum()
    grouped = grouped[~grouped.index.isin(exclude)]
    if grouped.empty:
        return ("", 0.0)
    return (grouped.idxmax(), float(grouped.max()))


def max_sector_share(sectors: pd.DataFrame) -> tuple:
    """Biggest single sector, e.g. ('Information Technology', 0.41).

    Takes the output of lookthrough.sector_exposure. Bonds, gold and the
    funds' unclassified small holdings are not sectors, so they never count.
    Because small holdings are unclassified, the true share can only be higher.
    """
    return _max_share(sectors, "sector", NOT_A_SECTOR)


def max_country_share(countries: pd.DataFrame) -> tuple:
    """Biggest single country, e.g. ('United States', 0.78).

    Takes the output of lookthrough.country_exposure (pass it the universe so
    single-country funds' small holdings count towards their country).
    """
    return _max_share(countries, "country", NOT_A_COUNTRY, COUNTRY_ALIASES)


def apparent_vs_real(portfolio: pd.DataFrame, holdings: pd.DataFrame,
                     universe: pd.DataFrame | None = None) -> dict:
    """How much overlap between funds shrinks the number of separate bets.

    'Apparent' pretends every fund's companies are different from every other
    fund's (Apple in VWRL and Apple in VUSA counted as two bets). 'Real' adds
    them together first, as look-through does. The gap is the hidden overlap.

    Returns: n_lines (rows in the CSV), n_eff_apparent, n_eff_real,
             overlap_loss (share of apparent bets lost to overlap, 0-1).
    """
    lines = explode(portfolio, holdings, universe)
    pieces = lines.loc[lines["kind"].isin([COMPANY, BLOCK]), "exposure"]
    apparent_hhi = float((pieces ** 2).sum())

    real = company_exposure(portfolio, holdings, universe)
    real_hhi = hhi(real)

    n_apparent = 1 / apparent_hhi if apparent_hhi > 0 else 0.0
    n_real = 1 / real_hhi if real_hhi > 0 else 0.0
    return {
        "n_lines": int(len(portfolio)),
        "n_eff_apparent": n_apparent,
        "n_eff_real": n_real,
        "overlap_loss": 1 - n_real / n_apparent if n_apparent > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Everything at once, for the pipeline and the app
# ---------------------------------------------------------------------------
@dataclass
class ConcentrationResult:
    hhi: float
    effective_n: float
    top10_share: float
    top_sector: str
    top_sector_share: float
    top_country: str
    top_country_share: float
    n_lines: int
    n_eff_apparent: float
    n_eff_real: float
    overlap_loss: float

    def to_dict(self) -> dict:
        return asdict(self)


def concentration_summary(portfolio: pd.DataFrame, holdings: pd.DataFrame,
                          universe: pd.DataFrame | None = None) -> ConcentrationResult:
    """Run every concentration measure on one portfolio."""
    companies = company_exposure(portfolio, holdings, universe)
    sector, sector_share = max_sector_share(sector_exposure(portfolio, holdings, universe))
    country, country_share = max_country_share(country_exposure(portfolio, holdings, universe))
    overlap = apparent_vs_real(portfolio, holdings, universe)
    return ConcentrationResult(
        hhi=hhi(companies),
        effective_n=effective_n(companies),
        top10_share=top_n_share(companies, TOP_N),
        top_sector=sector,
        top_sector_share=sector_share,
        top_country=country,
        top_country_share=country_share,
        **overlap,
    )