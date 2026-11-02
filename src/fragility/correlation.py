"""Correlation: do your holdings rise and fall together?

Plain idea: diversification is like carrying an umbrella AND sunglasses. It
only helps if the two are useful on different days. If everything you own
falls on the same bad weeks, owning lots of it doesn't protect you.

Correlation (rho) runs from -1 to +1:
   +1  always move together        (two S&P 500 funds)
    0  no connection               (shares vs gold, roughly)
   -1  always move opposite        (rare)

We use WEEKLY returns (Friday to Friday): UK and US markets close on
different holidays, and daily data would make them look less connected
than they really are.

All functions take weekly returns (prices.weekly_returns) or a correlation
matrix, plus weights as a Series indexed by ticker. No file reading here.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

from fragility.config import (
    CLUSTER_CUT,
    CORR_WINDOW_YEARS,
    MARKET_TICKER,
    MIN_HISTORY_YEARS,
    STRESS_WEEK_THRESHOLD,
)
from fragility.prices import weekly_returns

MIN_WEEKS = 26          # need at least half a year of overlapping weeks for a correlation
MIN_STRESS_WEEKS = 10   # need at least 10 bad weeks for a stress correlation


# ---------------------------------------------------------------------------
# Correlation matrices
# ---------------------------------------------------------------------------
def last_years(returns: pd.DataFrame, years: float | None) -> pd.DataFrame:
    """Keep only the last `years` of rows (None = keep everything)."""
    if years is None or returns.empty:
        return returns
    start = returns.index.max() - pd.DateOffset(years=years)
    return returns[returns.index > start]


def corr_matrix(returns: pd.DataFrame, years: float | None = CORR_WINDOW_YEARS,
                min_weeks: int = MIN_WEEKS) -> pd.DataFrame:
    """Normal-times correlation over the last `years` of weekly returns.

    Each pair uses the weeks both have data for. A pair with fewer than
    `min_weeks` shared weeks comes back as NaN (not enough evidence).
    """
    return last_years(returns, years).corr(min_periods=min_weeks)


def stress_weeks(returns: pd.DataFrame, market: str = MARKET_TICKER,
                 threshold: float = STRESS_WEEK_THRESHOLD) -> pd.Series:
    """True for weeks when the market fell by more than `threshold` (e.g. -2%)."""
    if market not in returns.columns:
        raise KeyError(f"Market ticker {market} is missing from the returns; "
                       "load its prices too.")
    return returns[market] < threshold


def stress_corr_matrix(returns: pd.DataFrame, market: str = MARKET_TICKER,
                       threshold: float = STRESS_WEEK_THRESHOLD, years: float | None = None,
                       min_weeks: int = MIN_STRESS_WEEKS) -> pd.DataFrame:
    """Correlation using ONLY the weeks when the global market fell more than 2%.

    This is the honest number: diversification matters most on bad weeks,
    and that is exactly when holdings tend to fall together. Uses the whole
    history by default (2015 onwards), because bad weeks are rare: a 3-year
    window holds only a handful of them.

    Caveat: looking at a narrow slice of weeks pulls correlations down a little
    for purely statistical reasons, so if anything this number is cautious.
    """
    window = last_years(returns, years)
    bad = window[stress_weeks(window, market, threshold)]
    return bad.corr(min_periods=min_weeks)


# ---------------------------------------------------------------------------
# One-number summaries
# ---------------------------------------------------------------------------
def _aligned(corr: pd.DataFrame, weights: pd.Series) -> tuple:
    tickers = [t for t in weights.index if t in corr.index]
    w = weights.reindex(tickers).astype(float)
    return corr.loc[tickers, tickers], w / w.sum() if w.sum() > 0 else w


def weighted_avg_corr(corr: pd.DataFrame, weights: pd.Series) -> float:
    """Average correlation between different holdings, big holdings counting more.

        sum over pairs i != j of  w_i * w_j * rho_ij
        --------------------------------------------
        sum over pairs i != j of  w_i * w_j

    Pairs without enough data (NaN) are skipped. One holding -> NaN.
    """
    c, w = _aligned(corr, weights)
    pair_weight = np.outer(w, w)
    np.fill_diagonal(pair_weight, 0.0)
    values = c.to_numpy(dtype=float)
    known = ~np.isnan(values)
    total = (pair_weight * known).sum()
    if total == 0:
        return float("nan")
    return float((pair_weight * np.where(known, values, 0.0)).sum() / total)


def diversification_ratio(returns: pd.DataFrame, weights: pd.Series,
                          years: float | None = CORR_WINDOW_YEARS) -> float:
    """How much mixing holdings calms the ride.

        (sum of each holding's own wobble, weighted)  /  (wobble of the whole portfolio)

    1.0 = no benefit at all (everything moves as one).
    1.4 = the mix is about 30% calmer than its parts.
    Uses weeks where every holding has a price, within the window.
    """
    tickers = [t for t in weights.index if t in returns.columns]
    window = last_years(returns[tickers], years).dropna()
    if len(window) < MIN_WEEKS or not tickers:
        return float("nan")
    w = weights.reindex(tickers).astype(float)
    w = w / w.sum()
    vols = window.std()
    portfolio_vol = float(np.sqrt(w @ window.cov() @ w))
    return float((w * vols).sum() / portfolio_vol) if portfolio_vol > 0 else float("nan")


# ---------------------------------------------------------------------------
# Clusters: groups that move together
# ---------------------------------------------------------------------------
def corr_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Turn correlation into distance: rho 1 -> 0 (same place), rho -1 -> 1 (far apart).

        d = sqrt(0.5 * (1 - rho))

    Missing correlations are treated as 0 (unknown = assume unrelated).
    """
    rho = corr.fillna(0.0).clip(-1, 1).to_numpy(dtype=float, copy=True)
    d = np.sqrt(0.5 * (1 - rho))
    np.fill_diagonal(d, 0.0)
    return pd.DataFrame(d, index=corr.index, columns=corr.columns)


def clusters(corr: pd.DataFrame, cut: float = CLUSTER_CUT,
             weights: pd.Series | None = None) -> list:
    """Groups of holdings that move together, biggest group first.

    Like seating guests at tables: anyone whose average distance to a table
    is under `cut` sits there. cut = 0.30 means roughly rho above 0.82.
    Returns a list of lists, e.g. [['VUSA.L', 'CSPX.L', 'EQQQ.L', 'AAPL'], ['SGLN.L']].
    """
    tickers = list(corr.index)
    if len(tickers) == 0:
        return []
    if len(tickers) == 1:
        return [tickers]

    condensed = squareform(corr_distance(corr).to_numpy(), checks=False)
    labels = fcluster(linkage(condensed, method="average"), t=cut, criterion="distance")

    groups = {}
    for ticker, label in zip(tickers, labels, strict=True):
        groups.setdefault(label, []).append(ticker)

    w = weights if weights is not None else pd.Series(1.0, index=tickers)
    size = {k: float(w.reindex(v).fillna(0).sum()) for k, v in groups.items()}
    ordered = sorted(groups, key=lambda k: (-size[k], groups[k][0]))
    return [sorted(groups[k], key=lambda t: -float(w.get(t, 0))) for k in ordered]


def effective_groups(groups: list, weights: pd.Series) -> float:
    """How many separate groups the money behaves like: 1 / sum(group weight^2).

    Same maths as effective_n in concentration.py, but on groups that move
    together instead of companies. 80% / 20% in two groups -> 1.47.
    """
    w = weights.astype(float) / weights.astype(float).sum()
    shares = [float(w.reindex(g).fillna(0).sum()) for g in groups]
    total = sum(s ** 2 for s in shares)
    return 1 / total if total > 0 else 0.0


def cluster_order(corr: pd.DataFrame) -> list:
    """Tickers ordered so that similar ones sit next to each other (for heatmaps)."""
    tickers = list(corr.index)
    if len(tickers) < 3:
        return tickers
    condensed = squareform(corr_distance(corr).to_numpy(), checks=False)
    return [tickers[i] for i in leaves_list(linkage(condensed, method="average"))]


def short_history(prices: pd.DataFrame, min_years: float = MIN_HISTORY_YEARS) -> list:
    """Tickers with less than `min_years` of prices: their numbers are less reliable."""
    out = []
    for t in prices.columns:
        s = prices[t].dropna()
        if s.empty or (s.index[-1] - s.index[0]).days < 365.25 * min_years:
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# Everything at once, for the pipeline and the app
# ---------------------------------------------------------------------------
@dataclass
class CorrelationResult:
    avg_corr: float                 # normal times, last 3 years
    avg_stress_corr: float          # bad market weeks only
    diversification_ratio: float
    groups: list                    # holdings that move together
    effective_groups: float         # how many separate groups the money behaves like
    n_stress_weeks: int
    short_history: list
    missing_prices: list
    corr: pd.DataFrame = field(repr=False)
    stress_corr: pd.DataFrame = field(repr=False)


def correlation_summary(prices: pd.DataFrame, weights: pd.Series,
                        market: str = MARKET_TICKER) -> CorrelationResult:
    """Run every correlation measure.

    prices:  daily GBP prices (prices.load_prices) for the held tickers AND the market.
    weights: portfolio weight of each holding, indexed by ticker.
    Holdings with no prices are left out and listed in `missing_prices`.
    """
    held = [t for t in weights.index if t in prices.columns and prices[t].notna().any()]
    missing = [t for t in weights.index if t not in held]
    w = weights.reindex(held).astype(float)
    w = w / w.sum() if w.sum() > 0 else w

    weekly = weekly_returns(prices[list(dict.fromkeys(held + [market]))])
    held_weekly = weekly[held]

    corr = corr_matrix(held_weekly)
    stress = stress_corr_matrix(weekly, market).reindex(index=held, columns=held)
    # Clusters use normal-times correlation: it rests on ~150 weeks, while the
    # stress figure rests on far fewer, so it is too jumpy to group holdings by.
    groups = clusters(corr, weights=w)

    return CorrelationResult(
        avg_corr=weighted_avg_corr(corr, w),
        avg_stress_corr=weighted_avg_corr(stress, w),
        diversification_ratio=diversification_ratio(held_weekly, w),
        groups=groups,
        effective_groups=effective_groups(groups, w),
        n_stress_weeks=int(stress_weeks(weekly, market).sum()),
        short_history=short_history(prices[held]),
        missing_prices=missing,
        corr=corr,
        stress_corr=stress,
    )