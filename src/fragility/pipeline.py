"""The pipeline: one call runs every module and returns one Report.

Plain idea: like a hospital check-up. The patient (a portfolio) goes through
each department in turn (look-through, concentration, correlation, crash
replay, scoring), and every result goes into one folder: the Report. The
Streamlit app only ever reads that folder; it never does maths itself.

    report = analyse_csv("my_portfolio.csv")      # from a file or an upload
    report = run_analysis(portfolio_df)           # from a clean table

Inputs (universe, holdings, prices, scenarios) can be passed in, which is how
the tests run without the internet. Anything not passed is loaded from disk
or downloaded.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fragility.alerts import build_alerts
from fragility.concentration import ConcentrationResult, concentration_summary
from fragility.config import MARKET_TICKER, PRICE_CACHE_MAX_AGE_DAYS
from fragility.correlation import CorrelationResult, correlation_summary
from fragility.lookthrough import (
    BLOCK,
    company_exposure,
    country_exposure,
    explode,
    overlap_matrix,
    sector_exposure,
)
from fragility.portfolio import load_portfolio
from fragility.prices import load_prices
from fragility.score import ScoreResult, fragility_score, metrics_from
from fragility.stress import (
    ScenarioResult,
    load_scenarios,
    replay_all,
    run_hypotheticals,
    scenario_table,
    tickers_needed,
    worst_scenario,
)
from fragility.universe import load_holdings, load_universe, share_symbols


@dataclass
class Report:
    """Everything the app shows, in one place."""

    # What was analysed
    portfolio: pd.DataFrame
    total_value: float
    errors: list = field(default_factory=list)      # rows rejected from the CSV
    notices: list = field(default_factory=list)     # things fixed for the user
    warnings: list = field(default_factory=list)    # caveats about the results

    # Look-through
    companies: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    lines: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)  # per route
    sectors: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    countries: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    overlap: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    # Measures
    concentration: ConcentrationResult | None = None
    correlation: CorrelationResult | None = None
    scenarios: list = field(default_factory=list, repr=False)   # ScenarioResult per crash
    scenario_table: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    worst: ScenarioResult | None = None
    hypotheticals: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    # Verdict
    score: ScoreResult | None = None
    alerts: list = field(default_factory=list)      # alerts.Alert, most serious first

    # Housekeeping
    holdings_as_of: str = ""
    prices_as_of: str = ""
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        """True when there was something to analyse and it produced a score."""
        return self.score is not None and not np.isnan(self.score.score)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _empty_correlation(weights: pd.Series) -> CorrelationResult:
    nan = float("nan")
    return CorrelationResult(nan, nan, nan, [list(weights.index)], nan, 0, [], [],
                             pd.DataFrame(), pd.DataFrame())


def _holdings_as_of(portfolio: pd.DataFrame, universe: pd.DataFrame) -> str:
    """Oldest holdings snapshot among the funds held, e.g. '2026-09-30'."""
    if "holdings_as_of" not in universe.columns:
        return ""
    held = universe[universe["ticker"].isin(portfolio["ticker"])]
    dates = [d for d in held["holdings_as_of"].fillna("") if d]
    return min(dates) if dates else ""


def _collect_warnings(companies, correlation, scenarios) -> list:
    """Plain-English caveats, each said once."""
    out = []
    not_looked = companies[(companies["kind"] == BLOCK)
                           & companies["company"].str.contains("not looked through")]
    for name in not_looked["company"]:
        out.append(f"{name.replace(' (not looked through)', '')}: no holdings data, "
                   "so it is treated as one single investment.")
    for t in correlation.missing_prices:
        out.append(f"{t}: no price history, so it is left out of the correlation "
                   "and crash results.")
    for t in correlation.short_history:
        out.append(f"{t}: under 2 years of prices, so its numbers are less reliable.")

    estimated = {}
    for r in scenarios:
        for t in r.estimated:
            estimated.setdefault(t, []).append(r.name)
    for t, names in estimated.items():
        out.append(f"{t} didn't exist yet in {', '.join(names)}: a similar older fund "
                   "stands in, so those results are estimates.")
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
def run_analysis(portfolio: pd.DataFrame, *,
                 universe: pd.DataFrame | None = None,
                 holdings: pd.DataFrame | None = None,
                 prices: pd.DataFrame | None = None,
                 scenarios: pd.DataFrame | None = None,
                 market: str = MARKET_TICKER,
                 max_price_age_days: float = PRICE_CACHE_MAX_AGE_DAYS) -> Report:
    """Run every module on a clean portfolio table (portfolio.py's output)."""
    started = time.perf_counter()
    if portfolio is None or portfolio.empty:
        raise ValueError("The portfolio is empty: nothing to analyse.")

    universe = load_universe() if universe is None else universe
    holdings = load_holdings() if holdings is None else holdings
    scenarios = load_scenarios() if scenarios is None else scenarios
    if prices is None:
        needed = tickers_needed(portfolio, universe) + [market]
        prices = load_prices(list(dict.fromkeys(needed)), max_age_days=max_price_age_days)

    weights = portfolio.set_index("ticker")["weight"].astype(float)
    total = float(portfolio["value_gbp"].sum())

    # 1. Look-through
    companies = company_exposure(portfolio, holdings, universe)
    lines = explode(portfolio, holdings, universe)      # which fund each piece comes via
    sectors = sector_exposure(portfolio, holdings, universe)
    countries = country_exposure(portfolio, holdings, universe)
    funds = portfolio.loc[portfolio["type"] == "etf", "ticker"].tolist()
    overlap = overlap_matrix(funds, holdings) if funds else pd.DataFrame()

    # 2. Concentration
    concentration = concentration_summary(portfolio, holdings, universe)

    # 3. Correlation (needs the market's prices to find the bad weeks)
    warnings = []
    if market in prices.columns and prices[market].notna().any():
        correlation = correlation_summary(prices, weights, market)
    else:
        correlation = _empty_correlation(weights)
        warnings.append(f"No prices for the market benchmark {market}, so the "
                        "'falls together' check was skipped.")

    # 4. Crash replay and what-if shocks
    results = replay_all(portfolio, prices, scenarios, universe)
    worst = worst_scenario(results)
    hypotheticals = run_hypotheticals(portfolio, holdings, universe)

    # 5. Score
    metrics = metrics_from(concentration, correlation,
                           worst.worst_loss if worst is not None else None)
    score = fragility_score(metrics)

    warnings += _collect_warnings(companies, correlation, results)
    last_price = prices.dropna(how="all").index.max() if not prices.empty else None

    report = Report(
        portfolio=portfolio, total_value=total, warnings=warnings,
        companies=companies, lines=lines, sectors=sectors, countries=countries, overlap=overlap,
        concentration=concentration, correlation=correlation,
        scenarios=results, scenario_table=scenario_table(results), worst=worst,
        hypotheticals=hypotheticals, score=score,
        holdings_as_of=_holdings_as_of(portfolio, universe),
        prices_as_of=last_price.date().isoformat() if last_price is not None else "",
    )
    report.alerts = build_alerts(report)
    report.seconds = time.perf_counter() - started
    return report


def analyse_csv(source, **kwargs) -> Report:
    """Read a portfolio CSV (path or uploaded file), check it, and run everything.

    Rows the CSV check rejects are reported in `errors`; if nothing valid is
    left, the Report comes back without a score (report.ok is False).
    Extra keyword arguments are passed to run_analysis. Pass share_lookup to
    replace the Yahoo check for individual shares (the tests do this).
    """
    universe = kwargs.get("universe")
    if universe is None:
        universe = kwargs["universe"] = load_universe()
    share_lookup = kwargs.pop("share_lookup", None)
    extra = {"share_lookup": share_lookup} if share_lookup else {}
    if kwargs.get("holdings") is not None:
        extra["share_symbols"] = share_symbols(kwargs["holdings"])

    checked = load_portfolio(source, universe, **extra)
    if not checked.ok:
        return Report(portfolio=checked.portfolio, total_value=0.0,
                      errors=checked.errors, notices=checked.notices)

    report = run_analysis(checked.portfolio, **kwargs)
    report.errors, report.notices = checked.errors, checked.notices
    return report