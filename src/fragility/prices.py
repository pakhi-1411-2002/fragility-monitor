"""Daily prices in GBP: download, fix pence, convert currency, clean, cache.

The one function the rest of the project calls is `load_prices(tickers)`.
It returns a table with one row per day and one column per ticker, every
price in pounds, cleaned, and cached on disk so repeat calls are instant.
"""

import logging
import time

import pandas as pd
import yfinance as yf

from fragility.config import (
    BASE_CURRENCY,
    CORR_FREQ,
    ETF_UNIVERSE_CSV,
    FFILL_LIMIT_DAYS,
    MAX_MISSING_SHARE,
    PRICE_CACHE,
    PRICE_CACHE_MAX_AGE_DAYS,
    START_DATE,
)

log = logging.getLogger(__name__)

PENCE = "GBp"           # yfinance's code for prices quoted in pence
PENCE_JUMP = 50         # a day-to-day ratio above this (or below 1/this) is a pence glitch


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------
def _yf_download(tickers: list, start: str) -> pd.DataFrame:
    """Thin wrapper so tests can replace the network call."""
    return yf.download(tickers, start=start, auto_adjust=True, progress=False,
                       threads=True)


def download_prices(tickers: list, start: str = START_DATE) -> pd.DataFrame:
    """Raw daily closes in each ticker's own currency. Index = date, columns = tickers."""
    tickers = sorted(set(tickers))
    if not tickers:
        return pd.DataFrame()

    raw = _yf_download(tickers, start)
    if raw is None or raw.empty:
        raise RuntimeError(f"No price data returned for {tickers}. "
                           "Check your internet connection, or wait if Yahoo is rate-limiting.")

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame()
    if list(close.columns) == ["Close"]:          # single ticker comes back unnamed
        close.columns = tickers

    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close.reindex(columns=tickers)        # tickers with no data -> all-NaN column
    return close.sort_index()


# ---------------------------------------------------------------------------
# 2. Currencies
# ---------------------------------------------------------------------------
def guess_currency(ticker: str) -> str:
    """Fallback when we can't ask Yahoo: London lines quote in pence, US shares in USD."""
    if ticker.endswith(".L"):
        return PENCE
    if "." not in ticker and "=" not in ticker:
        return "USD"
    return BASE_CURRENCY


def get_currencies(tickers: list) -> dict:
    """Currency of each ticker: etf_universe.csv where known, USD for plain US
    symbols, otherwise ask Yahoo (with a guess as fallback)."""
    known = {}
    if ETF_UNIVERSE_CSV.exists():
        universe = pd.read_csv(ETF_UNIVERSE_CSV, dtype=str).fillna("")
        known = {t: c for t, c in zip(universe["ticker"], universe["currency"], strict=True) if c}

    currencies = {}
    for ticker in tickers:
        if ticker in known:
            currencies[ticker] = known[ticker]
            continue
        if "." not in ticker and "=" not in ticker:   # plain US symbol, e.g. AAPL
            currencies[ticker] = "USD"
            continue
        try:
            currencies[ticker] = yf.Ticker(ticker).fast_info["currency"] or guess_currency(ticker)
        except Exception:  # noqa: BLE001 - Yahoo metadata is flaky; fall back to a guess
            currencies[ticker] = guess_currency(ticker)
        time.sleep(0.2)
    return currencies


def fix_pence_glitches(series: pd.Series) -> pd.Series:
    """Repair days where Yahoo switches a London price between pounds and pence.

    Like a till that sometimes prints 13,800 instead of 138.00: the price
    jumps by almost exactly 100x in a single day and later jumps back. No real
    fund moves 50x in a day, so each such jump flips a scale factor. The scale
    used on most days is treated as correct.
    """
    s = series.dropna()
    if len(s) < 2:
        return series

    ratio = (s / s.shift(1)).iloc[1:]
    factor, factors = 1.0, [1.0]
    for r in ratio:
        if PENCE_JUMP < r < PENCE_JUMP * 4:            # jumped up ~100x -> pence
            factor /= 100
        elif 1 / (PENCE_JUMP * 4) < r < 1 / PENCE_JUMP:  # jumped down ~100x -> pounds
            factor *= 100
        factors.append(factor)

    factors = pd.Series(factors, index=s.index)
    normal = factors.mode().iloc[0]                     # the scale on most days
    fixed = s * (factors / normal)
    return fixed.reindex(series.index).rename(series.name)


def to_gbp(prices: pd.DataFrame, currencies: dict, fx: pd.DataFrame) -> pd.DataFrame:
    """Convert every column to pounds.

    - GBp (pence): divide by 100.
    - Other currencies: divide by the GBPxxx rate (e.g. GBPUSD = 1.27 dollars per pound).
    `fx` holds columns like 'GBPUSD=X', aligned on the same dates as `prices`.
    """
    out = prices.copy()
    for ticker in out.columns:
        currency = currencies.get(ticker, BASE_CURRENCY)
        if ticker.endswith(".L"):                 # only London lines have this glitch
            out[ticker] = fix_pence_glitches(out[ticker])
        if currency == PENCE:
            out[ticker] = out[ticker] / 100
        elif currency not in (BASE_CURRENCY, ""):
            pair = fx_ticker(currency)
            if pair not in fx.columns:
                raise KeyError(f"Missing exchange rate {pair} needed for {ticker}")
            rate = fx[pair].reindex(out.index).ffill()
            out[ticker] = out[ticker] / rate
    return out


def fx_ticker(currency: str) -> str:
    """'USD' -> 'GBPUSD=X' (Yahoo's code for how many USD one pound buys)."""
    return f"{BASE_CURRENCY}{currency.upper()}=X"


# ---------------------------------------------------------------------------
# 3. Cleaning
# ---------------------------------------------------------------------------
SPIKE_JUMP = 0.20       # a one-day move bigger than this is suspicious...
SPIKE_RETURN = 0.05     # ...if the next day is back within 5% of the day before


def remove_spikes(series: pd.Series) -> tuple:
    """Blank out one-day price spikes that undo themselves the next day.

    Like a till typo that's corrected straight away: the price leaps (or
    drops) by more than 20% and the very next day is back where it was.
    Real markets don't do that; data glitches do (e.g. Yahoo recording a
    London fund's dollar price for one day, a x1.34 jump). The blanked day is
    filled with the previous price later, like any other gap.
    Returns (cleaned series, list of dates removed).
    """
    s = series.copy()
    known = s.dropna()
    removed = []
    values = known.to_numpy()
    for i in range(1, len(values) - 1):
        before, day, after = values[i - 1], values[i], values[i + 1]
        if before <= 0:
            continue
        jumped = abs(day / before - 1) > SPIKE_JUMP
        came_back = abs(after / before - 1) <= SPIKE_RETURN
        if jumped and came_back:
            removed.append(known.index[i])
    if removed:
        s.loc[removed] = float("nan")
    return s, removed


def clean_prices(prices: pd.DataFrame) -> tuple:
    """Fill short gaps, drop broken tickers. Returns (clean prices, list of dropped tickers).

    - UK and US markets close on different holidays, so one market has a gap
      while the other trades. Short gaps (up to FFILL_LIMIT_DAYS) are filled
      with the last known price.
    - One-day spikes that reverse the next day are removed (remove_spikes).
    - A ticker missing more than MAX_MISSING_SHARE of its days (counted from
      its first price, so newer funds aren't punished) is dropped.
    """
    p = prices.sort_index()
    p = p[~p.index.duplicated(keep="last")]
    p = p.where(p > 0)                      # zero or negative prices are errors
    for ticker in p.columns:                # one-day glitches that undo themselves
        p[ticker], removed = remove_spikes(p[ticker])
        for day in removed:
            log.warning("Removed a one-day price spike for %s on %s", ticker, day.date())
    p = p.dropna(how="all")
    p = p.ffill(limit=FFILL_LIMIT_DAYS)

    dropped = []
    for ticker in p.columns:
        first = p[ticker].first_valid_index()
        if first is None:
            dropped.append(ticker)
            log.warning("Dropping %s: no prices at all", ticker)
            continue
        missing = p.loc[first:, ticker].isna().mean()
        if missing > MAX_MISSING_SHARE:
            dropped.append(ticker)
            log.warning("Dropping %s: %.0f%% of days missing", ticker, missing * 100)

    return p.drop(columns=dropped), dropped


# ---------------------------------------------------------------------------
# 4. Cache
# ---------------------------------------------------------------------------
def cache_age_days(path=None) -> float:
    path = path or PRICE_CACHE
    if not path.exists():
        return float("inf")
    return (time.time() - path.stat().st_mtime) / 86_400


def read_cache(path=None) -> pd.DataFrame:
    path = path or PRICE_CACHE
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def write_cache(prices: pd.DataFrame, path=None) -> None:
    path = path or PRICE_CACHE
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(path)


def fetch_gbp_prices(tickers: list, start: str = START_DATE) -> tuple:
    """Download, convert to GBP and clean. Returns (prices, dropped tickers)."""
    currencies = get_currencies(tickers)
    pairs = sorted({fx_ticker(c) for c in currencies.values()
                    if c not in (BASE_CURRENCY, PENCE, "")})

    raw = download_prices(list(tickers) + pairs, start)
    fx = raw[pairs] if pairs else pd.DataFrame(index=raw.index)
    gbp = to_gbp(raw[list(tickers)], currencies, fx)
    return clean_prices(gbp)


def load_prices(tickers: list, start: str = START_DATE,
                max_age_days: float = PRICE_CACHE_MAX_AGE_DAYS,
                refresh: bool = False) -> pd.DataFrame:
    """Prices in GBP for `tickers`, from the cache when it is fresh enough.

    - Cache fresh and has every ticker  -> read from disk, no internet needed.
    - Cache fresh but missing a ticker  -> download only the new ones and add them.
    - Cache older than max_age_days, or refresh=True -> download everything again.
    Tickers that can't be priced are left out (and logged), never silently zero.
    """
    tickers = sorted(set(tickers))
    cached = read_cache()
    stale = refresh or cache_age_days() > max_age_days

    if stale or cached.empty:
        wanted = sorted(set(tickers) | set(cached.columns))
        prices, _ = fetch_gbp_prices(wanted, start)
        write_cache(prices)
    else:
        missing = [t for t in tickers if t not in cached.columns]
        if missing:
            new, _ = fetch_gbp_prices(missing, start)
            prices = cached.join(new, how="outer")
            write_cache(prices)
        else:
            prices = cached

    available = [t for t in tickers if t in prices.columns]
    return prices.loc[prices.index >= pd.Timestamp(start), available]


# ---------------------------------------------------------------------------
# 5. Returns
# ---------------------------------------------------------------------------
def returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns: 0.01 means the price rose 1% that day."""
    return prices.pct_change(fill_method=None).iloc[1:]


def weekly_returns(prices: pd.DataFrame, freq: str = CORR_FREQ) -> pd.DataFrame:
    """Week-on-week returns (Friday to Friday). Smooths out holiday mismatches."""
    weekly = prices.resample(freq).last()
    return weekly.pct_change(fill_method=None).iloc[1:]