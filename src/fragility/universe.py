"""Load and validate the reference tables: the ETF universe and ETF holdings."""

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
    return pd.read_csv(path, dtype=str).fillna("")


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