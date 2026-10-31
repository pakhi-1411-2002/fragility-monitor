"""Check etf_holdings.csv: weights sum to 1 per ETF and no blank tickers.

Run from the repo root:
    python scripts/check_holdings.py
Exits with code 1 if anything fails, so it can also run before commits.
"""

import sys

from fragility.config import WEIGHT_SUM_TOLERANCE
from fragility.universe import (
    RESIDUAL_ID,
    load_holdings,
    load_universe,
    missing_tickers,
    validate_holdings,
)


def main() -> int:
    universe = load_universe()
    holdings = load_holdings()

    # Per-ETF summary
    is_company = holdings["holding_id"] != RESIDUAL_ID
    blank = is_company & (holdings["holding_ticker"].str.strip() == "")
    summary = holdings.assign(blank_ticker=blank).groupby("etf_ticker").agg(
        rows=("holding_id", "size"),
        weight_sum=("weight", "sum"),
        blank_tickers=("blank_ticker", "sum"),
    )

    print(f"{'ETF':<9}{'rows':>6}{'weight sum':>12}{'blank tickers':>15}   result")
    for etf, r in summary.iterrows():
        ok = abs(r.weight_sum - 1) <= WEIGHT_SUM_TOLERANCE
        print(f"{etf:<9}{int(r.rows):>6}{r.weight_sum:>12.4f}{int(r.blank_tickers):>15}   "
              f"{'ok' if ok else 'FAIL'}")

    no_ticker = missing_tickers(holdings)
    if not no_ticker.empty:
        print(f"\nWarning: {len(no_ticker)} companies have no ticker "
              "(fine unless someone may hold them directly; fix via name_map.csv):")
        for r in no_ticker.itertuples():
            print(f"  - {r.holding_id:<40} {r.max_weight:>7.2%}  in {r.etfs}")

    problems = validate_holdings(holdings, universe)
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(" -", p)
        print("\nRESULT: FAIL")
        return 1

    print(f"\nRESULT: PASS - every ETF has holdings and sums to 1.00 +/- "
          f"{WEIGHT_SUM_TOLERANCE}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())