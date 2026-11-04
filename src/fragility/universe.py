"""Load and validate the reference tables: the ETF universe and ETF holdings."""

import re

import pandas as pd

from fragility.config import (
    ETF_HOLDINGS_CSV,
    ETF_UNIVERSE_CSV,
    RESIDUAL_LABEL,
    WEIGHT_SUM_TOLERANCE,
)

RESIDUAL_ID = "RESIDUAL"


def load_universe(path=ETF_UNIVERSE_CSV) -> pd.DataFrame:
    """The list of supported ETFs (one row per fund)."""
    universe = pd.read_csv(path, dtype=str).fillna("")
    return universe.apply(lambda col: col.str.strip())   # remove stray spaces


def load_holdings(path=ETF_HOLDINGS_CSV) -> pd.DataFrame:
    """What each ETF holds: top companies plus one residual row per fund."""
    holdings = pd.read_csv(path, dtype=str).fillna("")
    holdings["weight"] = pd.to_numeric(holdings["weight"], errors="coerce")
    return holdings


def validate_holdings(holdings: pd.DataFrame, universe: pd.DataFrame,
                      tolerance: float = WEIGHT_SUM_TOLERANCE) -> list:
    """Return a list of problems in plain English. An empty list means all good.

    Checks:
      1. Every ETF in the universe has holdings, and no unknown ETFs appear.
      2. Weights are numbers between 0 and 1.
      3. Each ETF's weights add up to 1 (within the tolerance).
      4. No blank holding_id.
      5. No company listed twice in the same ETF.
      6. Each equity ETF has exactly one residual row.
    """
    problems = []
    h = holdings.copy()
    h["weight"] = pd.to_numeric(h["weight"], errors="coerce")
    is_residual = h["holding_id"] == RESIDUAL_ID

    # 1. Coverage
    expected, present = set(universe["ticker"]), set(h["etf_ticker"])
    for etf in sorted(expected - present):
        problems.append(f"{etf}: no holdings at all")
    for etf in sorted(present - expected):
        problems.append(f"{etf}: has holdings but is not in etf_universe.csv")

    # 2. Valid weights
    bad = h[h["weight"].isna() | (h["weight"] < 0) | (h["weight"] > 1)]
    for row in bad.itertuples():
        problems.append(f"{row.etf_ticker}: invalid weight {row.weight!r} "
                        f"for {row.holding_name}")

    # 3. Sums
    totals = h.groupby("etf_ticker")["weight"].sum()
    for etf, total in totals.items():
        if abs(total - 1) > tolerance:
            problems.append(f"{etf}: weights add up to {total:.4f}, not 1.00 "
                            f"(allowed: 1.00 +/- {tolerance})")

    # 4. Blanks
    for row in h[h["holding_id"].astype(str).str.strip() == ""].itertuples():
        problems.append(f"{row.etf_ticker}: blank holding_id for {row.holding_name}")

    # 5. Duplicates
    dupes = h[h.duplicated(["etf_ticker", "holding_id"], keep=False)]
    for (etf, hid), _ in dupes.groupby(["etf_ticker", "holding_id"]):
        problems.append(f"{etf}: {hid} appears more than once")

    # 6. One residual per equity ETF
    equity = universe.loc[universe["asset_class"] == "equity", "ticker"]
    residual_counts = h[is_residual].groupby("etf_ticker").size()
    for etf in equity:
        if etf in present and residual_counts.get(etf, 0) != 1:
            problems.append(f"{etf}: should have exactly one '{RESIDUAL_LABEL}' row, "
                            f"has {residual_counts.get(etf, 0)}")

    return problems


def missing_tickers(holdings: pd.DataFrame) -> pd.DataFrame:
    """Companies with no ticker (residual rows excluded). A warning, not a failure.

    Tickers only matter for linking a share someone owns directly (e.g. AAPL)
    to the same company inside a fund, and that link also works by name.
    Fill any that matter via the holding_ticker column of name_map.csv.
    """
    blank = ((holdings["holding_id"] != RESIDUAL_ID)
             & (holdings["holding_ticker"].astype(str).str.strip() == ""))
    return (holdings[blank]
            .groupby(["holding_id", "holding_name"], as_index=False)
            .agg(etfs=("etf_ticker", lambda s: ", ".join(sorted(s))),
                 max_weight=("weight", "max"))
            .sort_values("max_weight", ascending=False))


US_SYMBOL = re.compile(r"^[A-Z]{1,5}(-[A-Z])?$")         # AAPL, BRK-B
UK_SYMBOL = re.compile(r"^[A-Z0-9]{2,5}(-[A-Z])?$")      # HSBA, BT-A


def yahoo_symbol(ticker: str, country: str) -> str:
    """Turn an issuer's ticker into the symbol Yahoo uses, or '' if we can't.

    Issuers write tickers their own way: 'BRK/B' or 'BRK.B' for Berkshire,
    'RR.' for Rolls-Royce. Yahoo wants 'BRK-B' and 'RR.L'. Only US and UK
    companies are handled: those are the shares a UK investor most often
    holds directly. Other markets use local codes (e.g. 2330 for TSMC) that
    would need an exchange suffix we can't reliably infer.
    """
    t = str(ticker).strip().upper().rstrip(".").replace("/", "-").replace(".", "-")
    if country == "United States" and US_SYMBOL.match(t):
        return t
    if country == "United Kingdom" and UK_SYMBOL.match(t):
        return f"{t}.L"
    return ""


def share_symbols(holdings: pd.DataFrame, max_rank: int | None = None) -> pd.DataFrame:
    """Companies inside the funds that someone could also own directly, with Yahoo symbols.

    Residual rows and bond/gold single-block rows are skipped. Returns one row
    per company: holding_id, holding_name, holding_ticker, country, yahoo_symbol
    (blank when the company's market isn't supported).

    max_rank: only keep companies that are among the `max_rank` biggest holdings
    of at least one fund (used to keep the price download small).
    """
    companies = holdings[(holdings["holding_id"] != RESIDUAL_ID)
                         & (holdings["holding_id"] != holdings["etf_ticker"])]
    if max_rank is not None:
        rank = companies.groupby("etf_ticker")["weight"].rank(method="first",
                                                              ascending=False)
        companies = companies[rank <= max_rank]
    companies = companies.drop_duplicates("holding_id")[
        ["holding_id", "holding_name", "holding_ticker", "country"]].copy()
    companies["yahoo_symbol"] = [yahoo_symbol(t, c) for t, c in
                                 zip(companies["holding_ticker"], companies["country"],
                                     strict=True)]
    return companies.reset_index(drop=True)