"""Read and validate a user's portfolio CSV.

Input (one row per holding):

    ticker,value_gbp
    VWRL.L,12000
    EQQQ.L,5000
    AAPL,2500

Output of `parse_portfolio`: a PortfolioResult with
  - portfolio: DataFrame [ticker, name, type, value_gbp, weight, holding_id]
  - errors:    rows that were rejected, in plain English
  - notices:   things we fixed for the user (merged duplicates, added .L, ...)

Bad rows never crash the app: they are reported and the good rows are kept.
"""

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd
from pydantic import BaseModel, ValidationError, field_validator

from fragility.config import ETF_HOLDINGS_CSV
from fragility.universe import load_holdings
from fragility.universe import share_symbols as build_share_symbols

TICKER_COLUMNS = ["ticker", "symbol", "code", "epic"]
VALUE_COLUMNS = ["value_gbp", "value", "value (£)", "value (gbp)", "amount",
                 "amount_gbp", "market value", "market_value", "gbp"]
OUTPUT_COLUMNS = ["ticker", "name", "type", "value_gbp", "weight", "holding_id"]

TEMPLATE_CSV = "ticker,value_gbp\nVWRL.L,12000\nEQQQ.L,5000\nAAPL,2500\n"


# ---------------------------------------------------------------------------
# One row, validated by pydantic
# ---------------------------------------------------------------------------
class HoldingRow(BaseModel):
    """A single line of the CSV. Pydantic checks it and gives a clear error if wrong."""

    ticker: str
    value_gbp: float

    @field_validator("ticker", mode="before")
    @classmethod
    def clean_ticker(cls, v):
        text = "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)
        text = text.strip().upper()
        if not text:
            raise ValueError("ticker is empty")
        if not re.fullmatch(r"[A-Z0-9.\-=^/]{1,15}", text):
            raise ValueError(f"'{text}' doesn't look like a ticker")
        return text

    @field_validator("value_gbp", mode="before")
    @classmethod
    def clean_value(cls, v):
        # Accept '£1,200', ' 1200 ', 1200.0
        text = str(v).replace("£", "").replace(",", "").strip()
        if text in {"", "nan", "None"}:
            raise ValueError("value is empty")
        try:
            number = float(text)
        except ValueError:
            raise ValueError(f"value '{v}' is not a number") from None
        if not math.isfinite(number) or number <= 0:
            raise ValueError("value must be more than £0")
        return number


@dataclass
class PortfolioResult:
    portfolio: pd.DataFrame
    errors: list = field(default_factory=list)
    notices: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when at least one valid holding remains."""
        return not self.portfolio.empty


@dataclass
class ShareInfo:
    """What we learn about a share that isn't one of our ETFs."""

    name: str


ShareLookup = Callable[[str], ShareInfo | None]


# ---------------------------------------------------------------------------
# Reading the CSV
# ---------------------------------------------------------------------------
def read_portfolio_csv(source) -> pd.DataFrame:
    """Read a path or uploaded file. Raises ValueError with a friendly message."""
    try:
        raw = pd.read_csv(source, dtype=str, skip_blank_lines=True)
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty.") from None
    except Exception as e:  # noqa: BLE001 - any parse failure gets the same advice
        raise ValueError(f"Couldn't read the file as a CSV ({e}).") from None
    return raw


def find_column(columns, options) -> str | None:
    lookup = {str(c).strip().lower(): c for c in columns}
    return next((lookup[o] for o in options if o in lookup), None)


# ---------------------------------------------------------------------------
# Matching shares to companies inside the funds
# ---------------------------------------------------------------------------
def simple_name(name: str) -> str:
    """'Apple Inc.' and 'APPLE INC' -> 'APPLE'. Used to link a share to a fund holding."""
    words = re.sub(r"[^A-Z0-9]+", " ", str(name).upper()).split()
    endings = {"INC", "CORP", "CORPORATION", "CO", "LTD", "LIMITED", "PLC", "SA", "AG",
               "NV", "SE", "HOLDINGS", "HOLDING", "GROUP", "THE", "CLASS", "A", "B", "C"}
    while words and words[-1] in endings:
        words.pop()
    return " ".join(words)


def link_to_holding(ticker: str, name: str, share_symbols: pd.DataFrame) -> str:
    """Find which company inside the funds this share is, so exposures add up.

    1. Same ticker (AAPL -> APPLE).  2. Same simplified name.  3. Otherwise the
    share stands alone and its own ticker becomes its id.
    """
    if share_symbols is not None and not share_symbols.empty:
        by_ticker = share_symbols[share_symbols["yahoo_symbol"] == ticker]
        if not by_ticker.empty:
            return by_ticker.iloc[0]["holding_id"]
        if name:
            target = simple_name(name)
            names = share_symbols["holding_name"].map(simple_name)
            by_name = share_symbols[names == target]
            if target and not by_name.empty:
                return by_name.iloc[0]["holding_id"]
    return ticker


def yahoo_share_lookup(ticker: str) -> ShareInfo | None:
    """Real lookup: does Yahoo have recent prices for this ticker? Returns its name if so."""
    import yfinance as yf

    try:
        t = yf.Ticker(ticker)
        if t.history(period="1mo").empty:
            return None
        try:
            info = t.info
            name = info.get("longName") or info.get("shortName") or ticker
        except Exception:  # noqa: BLE001 - name is a nice-to-have
            name = ticker
        return ShareInfo(name=name)
    except Exception:  # noqa: BLE001 - network or unknown symbol
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def parse_portfolio(raw: pd.DataFrame, universe: pd.DataFrame,
                    share_lookup: ShareLookup = yahoo_share_lookup,
                    share_symbols: pd.DataFrame | None = None) -> PortfolioResult:
    """Validate rows, classify each holding as an ETF or a share, compute weights."""
    errors, notices = [], []
    empty = pd.DataFrame(columns=OUTPUT_COLUMNS)

    ticker_col = find_column(raw.columns, TICKER_COLUMNS)
    value_col = find_column(raw.columns, VALUE_COLUMNS)
    if ticker_col is None or value_col is None:
        found = ", ".join(str(c) for c in raw.columns) or "none"
        errors.append("The CSV needs two columns: 'ticker' and 'value_gbp' "
                      f"(found: {found}). Download the template to see the format.")
        return PortfolioResult(empty, errors, notices)

    # 1. Validate each row (row numbers match what the user sees in Excel: header = row 1)
    rows = []
    for i, record in enumerate(raw[[ticker_col, value_col]].itertuples(index=False), start=2):
        ticker, value = record
        if all(pd.isna(x) or str(x).strip() == "" for x in (ticker, value)):
            continue  # fully blank line: skip quietly
        try:
            row = HoldingRow(ticker=ticker, value_gbp=value)
            rows.append({"row": i, "ticker": row.ticker, "value_gbp": row.value_gbp})
        except ValidationError as e:
            reasons = "; ".join(err["msg"].removeprefix("Value error, ") for err in e.errors())
            label = str(ticker).strip() if not pd.isna(ticker) and str(ticker).strip() else "?"
            errors.append(f"Row {i} ({label}): {reasons}")

    if not rows:
        if not errors:
            errors.append("The file has no holdings in it.")
        return PortfolioResult(empty, errors, notices)

    table = pd.DataFrame(rows)

    # 2. Classify: ETF from our list, or an individual share
    etfs = dict(zip(universe["ticker"], universe["name"], strict=True))
    resolved = {}
    for ticker in table["ticker"].unique():
        if ticker in etfs:
            resolved[ticker] = (ticker, etfs[ticker], "etf", ticker)
        elif f"{ticker}.L" in etfs:                       # typed VWRL instead of VWRL.L
            fixed = f"{ticker}.L"
            notices.append(f"Treated {ticker} as {fixed} (London listing).")
            resolved[ticker] = (fixed, etfs[fixed], "etf", fixed)
        else:
            info = share_lookup(ticker)
            if info is None:
                resolved[ticker] = None
            else:
                hid = link_to_holding(ticker, info.name, share_symbols)
                resolved[ticker] = (ticker, info.name, "share", hid)

    unknown = table[table["ticker"].map(lambda t: resolved[t] is None)]
    for r in unknown.itertuples():
        errors.append(f"Row {r.row} ({r.ticker}): not in our fund list and couldn't be "
                      "found as a share. Check the ticker (London shares end in .L, "
                      "e.g. HSBA.L).")
    table = table[table["ticker"].map(lambda t: resolved[t] is not None)].copy()
    if table.empty:
        return PortfolioResult(empty, errors, notices)

    parts = table["ticker"].map(resolved)
    table["ticker"] = parts.map(lambda p: p[0])
    table["name"] = parts.map(lambda p: p[1])
    table["type"] = parts.map(lambda p: p[2])
    table["holding_id"] = parts.map(lambda p: p[3])

    # 3. Merge duplicates (e.g. the same fund held in an ISA and a SIPP)
    counts = table["ticker"].value_counts()
    for ticker, n in counts[counts > 1].items():
        notices.append(f"{ticker} appeared {n} times; the values were added together.")
    table = (table.groupby("ticker", as_index=False, sort=False)
                  .agg(name=("name", "first"), type=("type", "first"),
                       value_gbp=("value_gbp", "sum"), holding_id=("holding_id", "first")))

    # 4. Weights
    table["weight"] = table["value_gbp"] / table["value_gbp"].sum()
    table = table.sort_values("value_gbp", ascending=False).reset_index(drop=True)
    return PortfolioResult(table[OUTPUT_COLUMNS], errors, notices)


def load_portfolio(source, universe: pd.DataFrame,
                   share_lookup: ShareLookup = yahoo_share_lookup,
                   share_symbols: pd.DataFrame | None = None) -> PortfolioResult:
    """Read a CSV (path or uploaded file) and validate it in one step.

    If share_symbols isn't given, it is built from etf_holdings.csv so that a
    directly held share (e.g. AAPL) is linked to the same company in the funds.
    """
    if share_symbols is None and ETF_HOLDINGS_CSV.exists():
        share_symbols = build_share_symbols(load_holdings())
    try:
        raw = read_portfolio_csv(source)
    except ValueError as e:
        return PortfolioResult(pd.DataFrame(columns=OUTPUT_COLUMNS), [str(e)], [])
    return parse_portfolio(raw, universe, share_lookup, share_symbols)
