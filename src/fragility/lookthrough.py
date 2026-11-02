"""Look-through: open every fund and add up what the portfolio really owns.

Plain idea: each fund is a box of chocolates. Open every box, count each kind
of chocolate across all boxes, and add any loose chocolates (shares held
directly). The maths for one company c:

    E_c = sum over holdings i of  w_i * h_ic

w_i  = share of the portfolio in holding i (from portfolio.py)
h_ic = share of fund i in company c (from etf_holdings.csv); a share held
       directly counts as 100% itself.

Every function takes plain DataFrames and returns DataFrames. No file reading,
no Streamlit, so everything is easy to test.
"""

import pandas as pd

from fragility.config import RESIDUAL_LABEL

RESIDUAL_ID = "RESIDUAL"
DIRECT = "direct"
UNCLASSIFIED = "Unclassified (smaller holdings)"
MIXED_COUNTRIES = "Mixed (smaller holdings)"

# kind of each looked-through line:
#   company  - a real company (counts towards concentration)
#   residual - a fund's "Other" bucket: many small companies, never one bet
#   block    - a bond fund, gold, or a fund with no holdings data: one asset
COMPANY, RESIDUAL, BLOCK = "company", "residual", "block"

# Funds whose small holdings must all come from one country.
REGION_COUNTRY = {"US": "United States", "UK": "United Kingdom", "Japan": "Japan"}

EXPLODED_COLUMNS = ["source", "holding_id", "holding_name", "sector", "country",
                    "kind", "exposure"]


# ---------------------------------------------------------------------------
# Core step: one row per (thing you hold, thing inside it)
# ---------------------------------------------------------------------------
def explode(portfolio: pd.DataFrame, holdings: pd.DataFrame,
            universe: pd.DataFrame | None = None) -> pd.DataFrame:
    """Open every fund. Returns one row per (portfolio line, underlying holding).

    portfolio: [ticker, type, weight, holding_id, name]  (from portfolio.py)
    holdings:  [etf_ticker, holding_id, holding_name, weight, sector, country]
    universe:  optional [ticker, region]; lets a single-country fund's
               residual be given that country (VUSA's small holdings are all US).
    """
    pieces = []
    fund_holdings = holdings.rename(columns={"weight": "fund_weight"})
    region = {} if universe is None else dict(zip(universe["ticker"], universe["region"],
                                                  strict=True))
    company_info = (holdings[holdings["holding_id"] != RESIDUAL_ID]
                    .drop_duplicates("holding_id")
                    .set_index("holding_id")[["holding_name", "sector", "country"]])

    for line in portfolio.itertuples(index=False):
        if line.type == "share":
            info = company_info.loc[line.holding_id] if line.holding_id in company_info.index \
                else None
            pieces.append(pd.DataFrame([{
                "source": DIRECT,
                "holding_id": line.holding_id,
                "holding_name": info["holding_name"] if info is not None else line.name,
                "sector": info["sector"] if info is not None else "Unknown",
                "country": info["country"] if info is not None else "Unknown",
                "kind": COMPANY,
                "exposure": line.weight,
            }]))
            continue

        inside = fund_holdings[fund_holdings["etf_ticker"] == line.ticker]
        if inside.empty:  # fund with no holdings data: keep it as one block
            pieces.append(pd.DataFrame([{
                "source": line.ticker, "holding_id": line.ticker,
                "holding_name": f"{line.name} (not looked through)",
                "sector": "Unknown", "country": "Unknown",
                "kind": BLOCK, "exposure": line.weight,
            }]))
            continue

        part = inside.assign(source=line.ticker, exposure=line.weight * inside["fund_weight"])
        is_residual = part["holding_id"] == RESIDUAL_ID
        is_block = part["holding_id"] == line.ticker          # bond/gold single block
        part["kind"] = COMPANY
        part.loc[is_residual, "kind"] = RESIDUAL
        part.loc[is_block, "kind"] = BLOCK
        part.loc[is_residual, "sector"] = UNCLASSIFIED
        part.loc[is_residual, "country"] = REGION_COUNTRY.get(region.get(line.ticker),
                                                              MIXED_COUNTRIES)
        pieces.append(part[EXPLODED_COLUMNS])

    if not pieces:
        return pd.DataFrame(columns=EXPLODED_COLUMNS)
    return pd.concat(pieces, ignore_index=True)


def _join_sources(sources: pd.Series) -> str:
    """'VWRL.L, EQQQ.L, direct': funds first (alphabetical), direct last."""
    unique = sorted(set(sources) - {DIRECT})
    return ", ".join(unique + ([DIRECT] if DIRECT in set(sources) else []))


def _total_value(portfolio: pd.DataFrame) -> float:
    return float(portfolio["value_gbp"].sum()) if "value_gbp" in portfolio else 0.0


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------
def company_exposure(portfolio: pd.DataFrame, holdings: pd.DataFrame,
                     universe: pd.DataFrame | None = None) -> pd.DataFrame:
    """What you really own, company by company, biggest first.

    Columns: holding_id, company, exposure (0-1), exposure_gbp, via, n_routes,
             sector, country, kind.
    All residual buckets are combined into ONE 'Other (residual)' row with
    kind='residual'; it is a crowd of small companies, never a single bet.
    """
    lines = explode(portfolio, holdings, universe)
    total = _total_value(portfolio)

    lines["group"] = lines["holding_id"].where(lines["kind"] != RESIDUAL, RESIDUAL_ID)
    out = (lines.groupby("group", as_index=False, sort=False)
                .agg(company=("holding_name", "first"),
                     exposure=("exposure", "sum"),
                     via=("source", _join_sources),
                     n_routes=("source", "nunique"),
                     sector=("sector", "first"),
                     country=("country", "first"),
                     kind=("kind", "first"))
                .rename(columns={"group": "holding_id"}))

    residual = out["kind"] == RESIDUAL
    out.loc[residual, ["company", "sector", "country"]] = [RESIDUAL_LABEL, UNCLASSIFIED,
                                                           MIXED_COUNTRIES]
    out["exposure_gbp"] = out["exposure"] * total
    # Biggest first; the residual bucket always goes last.
    out = out.assign(_last=residual).sort_values(["_last", "exposure"],
                                                 ascending=[True, False])
    return out[["holding_id", "company", "exposure", "exposure_gbp", "via", "n_routes",
                "sector", "country", "kind"]].reset_index(drop=True)


def _grouped_exposure(portfolio, holdings, universe, by: str) -> pd.DataFrame:
    lines = explode(portfolio, holdings, universe)
    total = _total_value(portfolio)
    out = (lines.groupby(by, as_index=False)
                .agg(exposure=("exposure", "sum"), via=("source", _join_sources))
                .sort_values("exposure", ascending=False)
                .reset_index(drop=True))
    out["exposure_gbp"] = out["exposure"] * total
    return out[[by, "exposure", "exposure_gbp", "via"]]


def sector_exposure(portfolio: pd.DataFrame, holdings: pd.DataFrame,
                    universe: pd.DataFrame | None = None) -> pd.DataFrame:
    """Share of money in each sector. Columns: sector, exposure, exposure_gbp, via.

    Funds' smaller holdings appear as 'Unclassified (smaller holdings)', so each
    named sector is a lower bound (the true share can only be higher).
    Bonds and gold appear as their own 'Bonds' / 'Commodities' rows.
    """
    return _grouped_exposure(portfolio, holdings, universe, "sector")


def country_exposure(portfolio: pd.DataFrame, holdings: pd.DataFrame,
                     universe: pd.DataFrame | None = None) -> pd.DataFrame:
    """Share of money in each country. Columns: country, exposure, exposure_gbp, via.

    Pass `universe` so single-country funds' smaller holdings count towards
    their country (e.g. all of VUSA is United States).
    """
    return _grouped_exposure(portfolio, holdings, universe, "country")


def overlap_matrix(etfs: list, holdings: pd.DataFrame) -> pd.DataFrame:
    """How much any two funds hold in common: 0 = nothing shared, 1 = identical.

    overlap(a, b) = sum over companies c of min(h_ac, h_bc)

    Like comparing two shopping baskets item by item and counting what's in
    both. A fund fully overlaps itself, so the diagonal is 1.
    Only named companies count; residual buckets are ignored, so this is
    a lower bound (with top-50 holdings, two S&P 500 funds score about 0.64,
    not ~1.0). Raise TOP_N_HOLDINGS in config.py for a fuller picture.
    """
    etfs = list(dict.fromkeys(etfs))   # unique, keep order
    named = holdings[(holdings["etf_ticker"].isin(etfs))
                     & (holdings["holding_id"] != RESIDUAL_ID)]
    table = named.pivot_table(index="holding_id", columns="etf_ticker", values="weight",
                              aggfunc="sum", fill_value=0.0).reindex(columns=etfs,
                                                                     fill_value=0.0)
    matrix = pd.DataFrame(0.0, index=etfs, columns=etfs)
    for a in etfs:
        for b in etfs:
            matrix.loc[a, b] = 1.0 if a == b else table[[a, b]].min(axis=1).sum()
    matrix.index.name = matrix.columns.name = None
    return matrix