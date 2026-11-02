"""Day 7: check the fragility score ranks five reference portfolios sensibly.

The five reference portfolios (£25,000 each):
  overlapping  - data/demo/overlapping.csv   (7 lines, one big US-tech bet)
  naive        - data/demo/naive.csv         (looks spread, leans UK/US)
  100% VWRL    - one global tracker
  diversified  - data/demo/diversified.csv   (shares, bonds, gold)
  60/40        - 60% VWRL, 40% VAGP (the classic shares/bonds mix)

Expected order (plan section 7.2):
  overlapping > naive > 100% VWRL > diversified, and diversified ~ 60/40
  (within MAX_GAP points of each other).

Workflow: run this, read the table, change ONE calm/alarm value in
config.py, run again, and log every change in the master plan (Section 13).
Change calm/alarm values before touching the weights.

Run from the repo root:
    python scripts/calibrate.py
Writes docs/calibration.csv (the numbers behind the write-up).
"""

import sys

import pandas as pd

from fragility.config import DEMO_DIR, MARKET_TICKER, ROOT, SCORE_COMPONENTS
from fragility.pipeline import run_analysis
from fragility.portfolio import load_portfolio
from fragility.prices import load_prices
from fragility.stress import load_scenarios, tickers_needed
from fragility.universe import load_holdings, load_universe

TOTAL = 25_000
MAX_GAP = 10                     # diversified and 60/40 should be within 10 points
EXPECTED_ORDER = ["overlapping", "naive", "100% VWRL", "diversified"]
OUTPUT = ROOT / "docs" / "calibration.csv"
SHORT = {"company_concentration": "top10", "sector_country_tilt": "tilt",
         "moves_together": "stress_corr", "crash_loss": "crash"}


def fund_portfolio(weights: dict, universe: pd.DataFrame) -> pd.DataFrame:
    """A portfolio of funds from {ticker: weight}, worth £25,000 in total."""
    names = dict(zip(universe["ticker"], universe["name"], strict=True))
    p = pd.DataFrame({"ticker": list(weights), "weight": list(weights.values())})
    p["value_gbp"] = p["weight"] * TOTAL
    p["type"], p["holding_id"] = "etf", p["ticker"]
    p["name"] = p["ticker"].map(names)
    return p


def reference_portfolios(universe: pd.DataFrame) -> dict:
    portfolios = {}
    for demo in ["overlapping", "naive"]:
        portfolios[demo] = load_portfolio(DEMO_DIR / f"{demo}.csv", universe).portfolio
    portfolios["100% VWRL"] = fund_portfolio({"VWRL.L": 1.0}, universe)
    portfolios["diversified"] = load_portfolio(DEMO_DIR / "diversified.csv",
                                               universe).portfolio
    portfolios["60/40"] = fund_portfolio({"VWRL.L": 0.6, "VAGP.L": 0.4}, universe)
    return portfolios


def score_table(portfolios: dict, **data) -> pd.DataFrame:
    """One row per portfolio: score, band, each measurement and its points."""
    rows = []
    for name, p in portfolios.items():
        report = run_analysis(p, **data)
        b = report.score.breakdown.set_index("component")
        row = {"portfolio": name, "score": report.score.score, "band": report.score.band,
               "main_reason": report.score.top_reason}
        for key, short in SHORT.items():
            row[short] = b.loc[key, "value"]
            row[f"{short}_pts"] = b.loc[key, "sub_score"]
        rows.append(row)
    return pd.DataFrame(rows)


def check_order(table: pd.DataFrame) -> list:
    """Problems with the ranking, in plain English. Empty list = all good."""
    s = table.set_index("portfolio")["score"]
    problems = []
    for higher, lower in zip(EXPECTED_ORDER, EXPECTED_ORDER[1:], strict=False):
        if not s[higher] > s[lower]:
            problems.append(f"{higher} ({s[higher]:.0f}) should score above "
                            f"{lower} ({s[lower]:.0f})")
    gap = abs(s["diversified"] - s["60/40"])
    if gap > MAX_GAP:
        problems.append(f"diversified ({s['diversified']:.0f}) and 60/40 "
                        f"({s['60/40']:.0f}) are {gap:.0f} points apart (max {MAX_GAP})")
    return problems


def main() -> int:
    universe, holdings, scenarios = load_universe(), load_holdings(), load_scenarios()
    portfolios = reference_portfolios(universe)

    needed = {MARKET_TICKER}
    for p in portfolios.values():
        needed |= set(tickers_needed(p, universe))
    prices = load_prices(sorted(needed))

    table = score_table(portfolios, universe=universe, holdings=holdings,
                        prices=prices, scenarios=scenarios)

    print("\nCurrent settings (config.py SCORE_COMPONENTS):")
    for key, c in SCORE_COMPONENTS.items():
        print(f"  {SHORT[key]:<12} calm {c['calm']:<5} alarm {c['alarm']:<5} "
              f"weight {c['weight']}")

    show = table.copy()
    for col in ("top10", "tilt", "crash"):
        show[col] = show[col].map(lambda v: f"{v:.0%}" if pd.notna(v) else "n/a")
    show["stress_corr"] = show["stress_corr"].map(
        lambda v: f"{v:.2f}" if pd.notna(v) else "n/a")
    for col in [c for c in show.columns if c.endswith("_pts")] + ["score"]:
        show[col] = show[col].map(lambda v: f"{v:.0f}" if pd.notna(v) else "-")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\n" + show.to_string(index=False))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    print(f"\nSaved {OUTPUT}")

    problems = check_order(table)
    if problems:
        print("\nRESULT: NOT YET")
        for p in problems:
            print("  -", p)
        print("\nChange ONE calm/alarm value in config.py, re-run, and log it.")
        return 1
    print("\nRESULT: PASS - overlapping > naive > 100% VWRL > diversified ~ 60/40")
    return 0


if __name__ == "__main__":
    sys.exit(main())