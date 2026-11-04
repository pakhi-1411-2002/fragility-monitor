"""Download and cache GBP prices for everything the app might need.

Pulls:
  1. every ETF in etf_universe.csv,
  2. US and UK companies among each fund's top SHARE_PRICE_TOP_N holdings
     (shares people often own directly, e.g. AAPL, HSBA.L),
  3. any individual shares in the demo portfolios (data/demo/*.csv).

Run from the repo root:
    python scripts/refresh_prices.py              # fresh download of everything
    python scripts/refresh_prices.py --use-cache  # only fetch what's missing
    python scripts/refresh_prices.py --etfs-only  # skip the company shares

Exits with code 1 if any ETF could not be priced (a missing share is only a warning).
"""

import argparse
import sys

import pandas as pd

from fragility.config import DEMO_DIR, PRICE_CACHE, SHARE_PRICE_TOP_N, START_DATE
from fragility.prices import load_prices
from fragility.universe import load_holdings, load_universe, share_symbols


def demo_share_tickers(etf_tickers: set) -> list:
    """Tickers in the demo portfolios that are not ETFs."""
    tickers = set()
    for path in sorted(DEMO_DIR.glob("*.csv")):
        demo = pd.read_csv(path, dtype=str)
        if "ticker" in demo.columns:
            tickers |= {t.strip() for t in demo["ticker"].dropna() if t.strip()}
    return sorted(tickers - etf_tickers)


def describe(prices: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """One line per ticker: first date, last date, latest price in GBP."""
    rows = []
    for t in tickers:
        if t not in prices.columns or prices[t].dropna().empty:
            rows.append({"ticker": t, "status": "MISSING"})
            continue
        s = prices[t].dropna()
        rows.append({"ticker": t, "status": "ok", "first": s.index[0].date(),
                     "last": s.index[-1].date(), "latest_gbp": round(float(s.iloc[-1]), 2)})
    return pd.DataFrame(rows, columns=["ticker", "status", "first", "last", "latest_gbp"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--use-cache", action="store_true",
                        help="keep cached prices and only download what's missing")
    parser.add_argument("--etfs-only", action="store_true",
                        help="price the ETFs only, skip individual company shares")
    args = parser.parse_args()

    universe = load_universe()
    etfs = sorted(universe["ticker"])

    shares = []
    if not args.etfs_only:
        symbols = share_symbols(load_holdings(), max_rank=SHARE_PRICE_TOP_N)
        supported = symbols[symbols["yahoo_symbol"] != ""]
        shares = sorted(set(supported["yahoo_symbol"]) | set(demo_share_tickers(set(etfs))))
        print(f"Companies in fund holdings: {len(symbols)} "
              f"({len(supported)} US/UK shares will be priced, "
              f"{len(symbols) - len(supported)} other markets skipped)")

    wanted = etfs + [s for s in shares if s not in etfs]
    print(f"Downloading {len(etfs)} ETFs + {len(shares)} shares since {START_DATE} "
          f"({'cache first' if args.use_cache else 'fresh download'})...")
    prices = load_prices(wanted, refresh=not args.use_cache)

    etf_report = describe(prices, etfs)
    share_report = describe(prices, shares)

    with pd.option_context("display.max_rows", None, "display.width", 120):
        print("\n=== ETFs ===")
        print(etf_report.to_string(index=False))

    missing_etfs = etf_report.loc[etf_report["status"] == "MISSING", "ticker"].tolist()
    missing_shares = share_report.loc[share_report["status"] == "MISSING", "ticker"].tolist()

    if shares:
        print(f"\n=== Shares: {len(shares) - len(missing_shares)} of {len(shares)} priced ===")
        if missing_shares:
            print("Could not price (fine unless someone holds them directly):")
            print("  " + ", ".join(missing_shares))

    print(f"\nCache: {PRICE_CACHE}  ({prices.shape[1]} tickers, "
          f"{prices.index.min().date()} to {prices.index.max().date()})")

    if missing_etfs:
        print("\nRESULT: FAIL - these ETFs have no prices: " + ", ".join(missing_etfs))
        print("Check the ticker in etf_universe.csv, or wait 10 minutes if Yahoo is rate-limiting.")
        return 1
    print("\nRESULT: PASS - every ETF is priced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())