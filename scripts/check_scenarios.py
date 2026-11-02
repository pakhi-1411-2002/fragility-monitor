"""Day 6: check the crash dates against real prices and write scenarios.csv.

For each crash in the plan we look at the reference fund's actual prices
(in GBP) around the planned dates and find:
  - the trough: the lowest close in the search range,
  - the peak:   the highest close BEFORE that trough.
Peak-to-trough is the worst fall someone could have lived through.

Two kinds of scenario:
  market - defined by the fall itself: dates are moved to the real peak/trough.
  event  - defined by a news event (Brexit vote, mini-budget): dates stay
           fixed, we only report what the reference did over those days.

Run from the repo root:
    python scripts/check_scenarios.py            # report + write scenarios.csv
    python scripts/check_scenarios.py --dry-run  # report only
"""

import argparse
from dataclasses import dataclass

import pandas as pd

from fragility.config import SCENARIOS_CSV
from fragility.prices import load_prices

SEARCH_DAYS = 30   # look this many days either side of the planned window


@dataclass
class Candidate:
    name: str
    start: str
    end: str
    reference: str
    kind: str          # "market" or "event"
    description: str


CANDIDATES = [
    Candidate("Covid crash", "2020-02-19", "2020-03-23", "VWRL.L", "market",
              "Fast global sell-off as Covid lockdowns began"),
    Candidate("2022 rate shock", "2022-01-03", "2022-10-12", "VWRL.L", "market",
              "Interest rates rose fast; shares and bonds fell together, tech hit hardest"),
    Candidate("Q4 2018 sell-off", "2018-10-01", "2018-12-24", "VWRL.L", "market",
              "US-led share sell-off on rate and trade-war fears"),
    Candidate("UK mini-budget", "2022-09-22", "2022-10-14", "IGLT.L", "event",
              "UK gilt crash after the September 2022 mini-budget"),
    Candidate("Brexit vote", "2016-06-23", "2016-06-27", "VWRL.L", "event",
              "Pound fell sharply after the EU referendum result"),
]


def find_peak_trough(prices: pd.Series, start: str, end: str,
                     search_days: int = SEARCH_DAYS) -> dict:
    """The worst peak-to-trough fall in [start - search_days, end + search_days].

    Walk through the days keeping track of the highest price so far; the
    biggest drop below that running high is the worst fall. Its peak is the
    high it fell from, its trough is the low it reached.
    """
    lo = pd.Timestamp(start) - pd.Timedelta(days=search_days)
    hi = pd.Timestamp(end) + pd.Timedelta(days=search_days)
    window = prices.dropna().loc[lo:hi]
    if window.empty:
        raise ValueError(f"No prices between {lo.date()} and {hi.date()}")

    drawdown = window / window.cummax() - 1
    trough_date = drawdown.idxmin()
    peak_date = window.loc[:trough_date][::-1].idxmax()   # last day at the high
    return {
        "peak_date": peak_date,
        "trough_date": trough_date,
        "fall": float(drawdown.min()),
    }


def move_between(prices: pd.Series, start: str, end: str) -> float:
    """Change from the last close on/before `start` to the last close on/before `end`."""
    s = prices.dropna()
    a = s.loc[:pd.Timestamp(start)]
    b = s.loc[:pd.Timestamp(end)]
    if a.empty or b.empty:
        return float("nan")
    return float(b.iloc[-1] / a.iloc[-1] - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = parser.parse_args()

    references = sorted({c.reference for c in CANDIDATES})
    prices = load_prices(references)

    rows, report, changes, warnings = [], [], [], []
    for c in CANDIDATES:
        series = prices[c.reference]
        planned_move = move_between(series, c.start, c.end)

        if c.kind == "market":
            found = find_peak_trough(series, c.start, c.end)
            start = found["peak_date"].date().isoformat()
            end = found["trough_date"].date().isoformat()
            fall = found["fall"]
            if fall > -0.05:
                warnings.append(f"{c.name}: worst fall near these dates is only {fall:+.1%}; "
                                "check the dates or the reference ticker")
            if (start, end) != (c.start, c.end):
                changes.append(f"{c.name}: {c.start} to {c.end} -> {start} to {end} "
                               f"({c.reference} peak/trough in GBP)")
        else:
            start, end, fall = c.start, c.end, planned_move

        report.append({
            "scenario": c.name, "kind": c.kind, "reference": c.reference,
            "planned": f"{c.start} to {c.end}", "planned_move": planned_move,
            "final": f"{start} to {end}", "final_move": fall,
            "trading_days": int(series.loc[start:end].dropna().shape[0]),
        })
        rows.append({"name": c.name, "start_date": start, "end_date": end,
                     "reference": c.reference, "kind": c.kind, "description": c.description})

    table = pd.DataFrame(report)
    for col in ("planned_move", "final_move"):
        table[col] = table[col].map(lambda v: f"{v:+.1%}")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print("\n=== Scenarios (moves are in GBP) ===")
        print(table.to_string(index=False))

    if changes:
        print("\nDates moved (add these to the change log, Section 13):")
        for line in changes:
            print(f"  - {line}")
    else:
        print("\nAll planned dates matched the real peaks and troughs.")

    for line in warnings:
        print(f"\nWARNING: {line}")

    if args.dry_run:
        print("\n--dry-run: scenarios.csv not written.")
        return
    SCENARIOS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(SCENARIOS_CSV, index=False)
    print(f"\nWrote {SCENARIOS_CSV}")


if __name__ == "__main__":
    main()