"""This is a script to check candidate ETF tickers on yfinance and write etf_universe.csv.

Run from the repo root:
    python scripts/check_universe.py
"""

import time

import pandas as pd
import yfinance as yf

from fragility.config import ETF_UNIVERSE_CSV

EARLIEST_SCENARIO = pd.Timestamp("2016-06-23")  # Brexit vote, our oldest stress test
MAX_STALE_DAYS = 10                              # last price older than this = dead ticker

# ticker, name, issuer, asset_class, region, proxy (fill proxy if history is short)
CANDIDATES = [
    ("VWRL.L", "Vanguard FTSE All-World", "Vanguard", "equity", "Global", ""),
    ("SWDA.L", "iShares Core MSCI World", "iShares", "equity", "Developed", ""),
    ("VEVE.L", "Vanguard FTSE Developed World", "Vanguard", "equity", "Developed", ""),
    ("HMWO.L", "HSBC MSCI World", "HSBC", "equity", "Developed", ""),
    ("VUSA.L", "Vanguard S&P 500 (Dist)", "Vanguard", "equity", "US", ""),
    ("CSPX.L", "iShares Core S&P 500", "iShares", "equity", "US", ""),
    ("VUAG.L", "Vanguard S&P 500 (Acc)", "Vanguard", "equity", "US", "VUSA.L"),
    ("EQQQ.L", "Invesco EQQQ Nasdaq-100", "Invesco", "equity", "US", ""),
    ("IITU.L", "iShares S&P 500 IT Sector", "iShares", "equity", "US", ""),
    ("INRG.L", "iShares Global Clean Energy", "iShares", "equity", "Global", ""),
    ("ISF.L", "iShares Core FTSE 100", "iShares", "equity", "UK", ""),
    ("VUKE.L", "Vanguard FTSE 100", "Vanguard", "equity", "UK", ""),
    ("VMID.L", "Vanguard FTSE 250", "Vanguard", "equity", "UK", ""),
    ("VERX.L", "Vanguard FTSE Dev Europe ex UK", "Vanguard", "equity", "Europe", ""),
    ("VJPN.L", "Vanguard FTSE Japan", "Vanguard", "equity", "Japan", ""),
    ("EIMI.L", "iShares Core MSCI EM IMI", "iShares", "equity", "Emerging", ""),
    ("VFEM.L", "Vanguard FTSE Emerging Markets", "Vanguard", "equity", "Emerging", ""),
    ("IGLT.L", "iShares Core UK Gilts", "iShares", "bond", "UK", ""),
    ("VAGP.L", "Vanguard Global Aggregate Bond GBP Hedged", "Vanguard", "bond", "Global", "IGLT.L"),
    ("SGLN.L", "iShares Physical Gold", "iShares", "commodity", "Global", ""),
]


def check(ticker: str, proxy: str) -> dict:
    """Ask yfinance about one ticker and return what we learned."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="max", auto_adjust=True)
        if hist.empty or hist["Close"].dropna().empty:
            return {"status": "FAIL: no price data"}

        hist = hist.dropna(subset=["Close"])
        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)
        first, last = hist.index[0], hist.index[-1]

        try:
            currency = t.fast_info["currency"]
        except Exception:
            currency = ""

        days_old = (pd.Timestamp.today().normalize() - last).days
        if days_old > MAX_STALE_DAYS:
            status = f"FAIL: last price {days_old} days old"
        elif first > EARLIEST_SCENARIO and not proxy:
            status = "OK: short history (needs proxy?)"
        else:
            status = "OK"

        return {
            "status": status,
            "currency": currency,
            "first_date": first.date(),
            "last_date": last.date(),
            "rows": len(hist),
            "last_close": round(float(hist["Close"].iloc[-1]), 2),
        }
    except Exception as e:  # network errors, rate limits, odd tickers
        return {"status": f"FAIL: {type(e).__name__}: {e}"}


def main() -> None:
    results = []
    for ticker, name, issuer, asset_class, region, proxy in CANDIDATES:
        print(f"Checking {ticker} ...")
        info = check(ticker, proxy)
        results.append({
            "ticker": ticker, "name": name, "issuer": issuer,
            "asset_class": asset_class, "region": region, "proxy": proxy, **info,
        })
        time.sleep(0.5)  # be polite to Yahoo, avoids rate limits

    report = pd.DataFrame(results)

    with pd.option_context("display.max_rows", None, "display.width", 200):
        print("\n=== Report ===")
        print(report[["ticker", "status", "currency", "first_date",
                      "last_date", "rows", "last_close"]].to_string(index=False))

    passed = report[report["status"].str.startswith("OK")].copy()
    passed["holdings_as_of"] = ""  # filled on Day 2
    cols = ["ticker", "name", "issuer", "asset_class", "region",
            "currency", "proxy", "first_date", "holdings_as_of"]
    ETF_UNIVERSE_CSV.parent.mkdir(parents=True, exist_ok=True)
    passed[cols].to_csv(ETF_UNIVERSE_CSV, index=False)

    print(f"\n{len(passed)} of {len(report)} tickers passed.")
    print(f"Wrote {ETF_UNIVERSE_CSV}")


if __name__ == "__main__":
    main()