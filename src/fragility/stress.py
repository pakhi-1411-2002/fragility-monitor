"""Stress tests: what would TODAY's portfolio have lost in past crashes?

Plain idea: take the portfolio as it is now, travel back in time, and hold
it through a past crash without touching it ("buy and hold"). Each holding
moves the way it really did; the portfolio moves by the weighted mix.

    portfolio return = sum over holdings i of  w_i * r_i

Plus two made-up ("hypothetical") shocks that use look-through instead of
history, e.g. "what if US tech fell 20% tomorrow?".

Holdings that didn't exist yet during a crash borrow their proxy's prices
(from etf_universe.csv) and are marked "estimated". Holdings with neither
are left out and listed, and the result says what share of the portfolio
it covers.

No file reading except load_scenarios; no Streamlit.
"""

from dataclasses import dataclass, field

import pandas as pd

from fragility.config import HYPOTHETICAL_SHOCKS, SCENARIOS_CSV
from fragility.lookthrough import BLOCK, explode


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_scenarios(path=None) -> pd.DataFrame:
    """The crash list written by scripts/check_scenarios.py."""
    return pd.read_csv(path or SCENARIOS_CSV, dtype=str).fillna("")


def proxy_map(universe: pd.DataFrame | None) -> dict:
    """{ticker: proxy} for funds that have one."""
    if universe is None or "proxy" not in universe.columns:
        return {}
    u = universe.fillna("")
    return {t: p for t, p in zip(u["ticker"], u["proxy"], strict=True) if p}


def tickers_needed(portfolio: pd.DataFrame, universe: pd.DataFrame | None = None) -> list:
    """Every ticker whose prices the stress tests need: holdings plus their proxies."""
    held = list(portfolio["ticker"])
    proxies = proxy_map(universe)
    return list(dict.fromkeys(held + [proxies[t] for t in held if t in proxies]))


# ---------------------------------------------------------------------------
# Peak-to-trough
# ---------------------------------------------------------------------------
def max_drawdown_in_window(values: pd.Series) -> dict:
    """The worst fall from a high point to a later low point.

    Like a hill walk: remember the highest point so far; the deepest dip
    below it is the worst fall. Returns fall (e.g. -0.32), peak_date, trough_date.
    """
    v = values.dropna()
    if v.empty:
        return {"fall": float("nan"), "peak_date": None, "trough_date": None}
    drawdown = v / v.cummax() - 1
    trough = drawdown.idxmin()
    peak = v.loc[:trough][::-1].idxmax()        # last day at that high
    return {"fall": float(drawdown.min()), "peak_date": peak, "trough_date": trough}


# ---------------------------------------------------------------------------
# Historical replay
# ---------------------------------------------------------------------------
@dataclass
class ScenarioResult:
    name: str
    kind: str
    start: pd.Timestamp
    end: pd.Timestamp
    portfolio_return: float          # start to end, e.g. -0.25
    loss_gbp: float                  # negative = money lost
    max_drawdown: float              # worst dip inside the window
    coverage: float                  # share of the portfolio included (0-1)
    estimated: list = field(default_factory=list)   # used a proxy
    excluded: list = field(default_factory=list)    # no prices at all
    contributors: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    path: pd.Series = field(default_factory=pd.Series, repr=False)  # value, start = 1.0

    @property
    def worst_loss(self) -> float:
        """The more painful of end-to-end return and the worst dip (as a negative number)."""
        return min(self.portfolio_return, self.max_drawdown)


def _price_on(series: pd.Series, date: pd.Timestamp) -> float | None:
    """Last known price on or before `date` (markets are shut at weekends)."""
    known = series.dropna().loc[:date]
    return float(known.iloc[-1]) if not known.empty else None


def _relative_path(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp):
    """Price path from start to end, scaled so the start is 1.0. None if no start price."""
    base = _price_on(series, start)
    if base is None or base <= 0:
        return None
    s = series.dropna()
    path = s.loc[(s.index > start) & (s.index <= end)]
    return pd.concat([pd.Series([base], index=[start]), path]) / base


def replay(portfolio: pd.DataFrame, prices: pd.DataFrame, scenario,
           universe: pd.DataFrame | None = None) -> ScenarioResult:
    """Hold today's portfolio through one past scenario.

    portfolio: [ticker, weight, value_gbp]   prices: daily GBP, one column per ticker
    scenario:  a row of scenarios.csv (name, start_date, end_date, kind)
    """
    start, end = pd.Timestamp(scenario["start_date"]), pd.Timestamp(scenario["end_date"])
    proxies = proxy_map(universe)
    total = float(portfolio["value_gbp"].sum()) if "value_gbp" in portfolio else 0.0

    paths, used_weights, estimated, excluded = {}, {}, [], []
    for line in portfolio.itertuples(index=False):
        path = None
        if line.ticker in prices.columns:
            path = _relative_path(prices[line.ticker], start, end)
        if path is None and line.ticker in proxies and proxies[line.ticker] in prices.columns:
            path = _relative_path(prices[proxies[line.ticker]], start, end)
            if path is not None:
                estimated.append(line.ticker)
        if path is None:
            excluded.append(line.ticker)
            continue
        paths[line.ticker] = path
        used_weights[line.ticker] = float(line.weight)

    coverage = sum(used_weights.values())
    if coverage == 0:
        return ScenarioResult(scenario["name"], scenario.get("kind", ""), start, end,
                              float("nan"), float("nan"), float("nan"), 0.0,
                              estimated, excluded)

    # Leave out what we can't price and spread its weight over the rest.
    w = pd.Series(used_weights) / coverage
    table = pd.DataFrame(paths).sort_index().ffill()
    value = (table * w).sum(axis=1)

    returns = table.iloc[-1] - 1
    contributors = pd.DataFrame({
        "ticker": returns.index,
        "weight": w.values,
        "return": returns.values,
        "contribution": (w * returns).values,             # share of the portfolio lost
    })
    contributors["loss_gbp"] = contributors["contribution"] * total
    contributors = contributors.sort_values("contribution").reset_index(drop=True)

    portfolio_return = float(value.iloc[-1] - 1)
    return ScenarioResult(
        name=scenario["name"], kind=scenario.get("kind", ""), start=start, end=end,
        portfolio_return=portfolio_return,
        loss_gbp=portfolio_return * total,
        max_drawdown=max_drawdown_in_window(value)["fall"],
        coverage=coverage, estimated=estimated, excluded=excluded,
        contributors=contributors, path=value,
    )


def replay_all(portfolio: pd.DataFrame, prices: pd.DataFrame, scenarios: pd.DataFrame,
               universe: pd.DataFrame | None = None) -> list:
    """Replay every scenario. Returns a list of ScenarioResult."""
    return [replay(portfolio, prices, row, universe) for _, row in scenarios.iterrows()]


def scenario_table(results: list, top: int = 3) -> pd.DataFrame:
    """One row per scenario for the app: % and £ loss, worst dip, worst holdings, flags."""
    rows = []
    for r in results:
        worst = r.contributors[r.contributors["contribution"] < 0].head(top)
        rows.append({
            "scenario": r.name,
            "dates": f"{r.start.date()} to {r.end.date()}",
            "return": r.portfolio_return,
            "loss_gbp": r.loss_gbp,
            "max_drawdown": r.max_drawdown,
            "worst_contributors": ", ".join(worst["ticker"]),
            "estimated": ", ".join(r.estimated),
            "excluded": ", ".join(r.excluded),
            "coverage": r.coverage,
        })
    return pd.DataFrame(rows)


def worst_scenario(results: list) -> ScenarioResult | None:
    """The scenario that would have hurt most (by worst_loss)."""
    valid = [r for r in results if pd.notna(r.portfolio_return)]
    return min(valid, key=lambda r: r.worst_loss) if valid else None


# ---------------------------------------------------------------------------
# Hypothetical shocks (look-through, no history)
# ---------------------------------------------------------------------------
def hypothetical_shock(lines: pd.DataFrame, shock: float, sector: str | None = None,
                       country: str | None = None) -> dict:
    """Loss if every company matching sector AND country fell by `shock`.

    lines: lookthrough.explode output. Bond and gold blocks are never hit
    (an equity shock doesn't apply to them). Funds' unclassified small
    holdings have no sector, so a sector shock skips them: a lower bound.
    Returns exposed (share of portfolio hit) and loss (share of portfolio, negative).
    """
    hit = lines["kind"] != BLOCK
    if sector:
        hit &= lines["sector"] == sector
    if country:
        hit &= lines["country"] == country
    exposed = float(lines.loc[hit, "exposure"].sum())
    return {"exposed": exposed, "loss": exposed * shock}


def run_hypotheticals(portfolio: pd.DataFrame, holdings: pd.DataFrame,
                      universe: pd.DataFrame | None = None,
                      shocks: list | None = None) -> pd.DataFrame:
    """Every configured shock. Columns: scenario, exposed, loss, loss_gbp."""
    lines = explode(portfolio, holdings, universe)
    total = float(portfolio["value_gbp"].sum()) if "value_gbp" in portfolio else 0.0
    rows = []
    for s in shocks if shocks is not None else HYPOTHETICAL_SHOCKS:
        out = hypothetical_shock(lines, s["shock"], s.get("sector"), s.get("country"))
        rows.append({"scenario": s["name"], "exposed": out["exposed"],
                     "loss": out["loss"], "loss_gbp": out["loss"] * total})
    return pd.DataFrame(rows, columns=["scenario", "exposed", "loss", "loss_gbp"])