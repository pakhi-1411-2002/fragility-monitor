"""Day 10: check the numbers against reality (plan section 10).

Unit tests prove the maths is right on made-up data. These checks prove the
REAL data flows through sensibly: each one compares a result with something
we know from outside the app (a fund's published figures, an index fact, a
real crash).

Run from the repo root:
    python scripts/sanity_checks.py
Any FAIL is a "surprise": log it in the plan's change log (Section 13), then fix it.
"""

import sys

import pandas as pd

from fragility.config import DEMO_DIR, MARKET_TICKER, TOP_N_HOLDINGS
from fragility.lookthrough import COMPANY, overlap_matrix  # noqa: F401 (used in query)
from fragility.pipeline import run_analysis
from fragility.portfolio import load_portfolio
from fragility.prices import load_prices
from fragility.stress import load_scenarios, tickers_needed
from fragility.universe import load_holdings, load_universe

TOTAL = 25_000
results = []


def check(name: str, expected: str, actual: str, ok: bool, hint: str = "") -> None:
    results.append({"check": name, "expected": expected, "actual": actual,
                    "result": "PASS" if ok else "FAIL", "hint": "" if ok else hint})


def one_fund(ticker: str, universe: pd.DataFrame) -> pd.DataFrame:
    name = universe.set_index("ticker").loc[ticker, "name"]
    return pd.DataFrame({"ticker": [ticker], "weight": [1.0], "value_gbp": [TOTAL],
                         "type": ["etf"], "holding_id": [ticker], "name": [name]})


def main() -> int:
    universe, holdings, scenarios = load_universe(), load_holdings(), load_scenarios()
    demos = {d: load_portfolio(DEMO_DIR / f"{d}.csv", universe).portfolio
             for d in ["overlapping", "naive", "diversified"]}
    singles = {t: one_fund(t, universe) for t in ["VWRL.L", "VUSA.L", "ISF.L", "IGLT.L"]}

    needed = {MARKET_TICKER} | set(universe["ticker"])
    for p in list(demos.values()) + list(singles.values()):
        needed |= set(tickers_needed(p, universe))
    prices = load_prices(sorted(needed))
    data = dict(universe=universe, holdings=holdings, prices=prices, scenarios=scenarios)
    run = {name: run_analysis(p, **data) for name, p in {**singles, **demos}.items()}

    # 1. 100% VWRL: its biggest company should equal the fund's own top holding.
    fund_top = holdings[(holdings["etf_ticker"] == "VWRL.L")
                        & (holdings["holding_id"] != "RESIDUAL")].nlargest(1, "weight").iloc[0]
    our_top = run["VWRL.L"].companies.query("kind == @COMPANY").iloc[0]
    check("100% VWRL: biggest company = fund's top holding",
          f"{fund_top['holding_name']} {fund_top['weight']:.2%} (compare with Vanguard's site)",
          f"{our_top['company']} {our_top['exposure']:.2%}",
          abs(our_top["exposure"] - fund_top["weight"]) < 0.001,
          "Look-through or name matching is changing a single fund's weights.")

    # 2. VUSA and CSPX track the same index (S&P 500), so overlap should be ~1.
    ov = overlap_matrix(["VUSA.L", "CSPX.L"], holdings).loc["VUSA.L", "CSPX.L"]
    check("VUSA vs CSPX overlap (same index)", ">= 95%", f"{ov:.0%}", ov >= 0.95,
          f"Only the top {TOP_N_HOLDINGS} holdings are compared. Raise TOP_N_HOLDINGS "
          "in config.py (e.g. 500) and re-run build_holdings.py.")

    # 3. 100% VWRL in the Covid crash should fall about as much as VWRL really did.
    covid = next(r for r in run["VWRL.L"].scenarios if r.name == "Covid crash")
    s = prices[MARKET_TICKER].dropna()
    real = float(s.loc[:covid.end].iloc[-1] / s.loc[:covid.start].iloc[-1] - 1)
    check("100% VWRL Covid replay = VWRL's real move", f"{real:+.1%} (about -20% to -30%)",
          f"{covid.portfolio_return:+.1%}",
          abs(covid.portfolio_return - real) < 0.005 and -0.35 < real < -0.15,
          "Replay maths or the scenario dates are off.")

    # 4. An S&P 500 fund's top 10 is a well-known figure: roughly a third or more.
    t10 = run["VUSA.L"].concentration.top10_share
    check("100% VUSA: top-10 share (S&P 500 top 10)", "30% to 45%", f"{t10:.0%}",
          0.30 <= t10 <= 0.45, "Holdings for VUSA may be incomplete or mis-scaled.")

    # 5. Single-country funds should be (almost) all that country.
    for ticker, country, floor in [("VUSA.L", "United States", 0.95),
                                   ("ISF.L", "United Kingdom", 0.85)]:
        c = run[ticker].concentration
        check(f"100% {ticker}: country", f"{country} >= {floor:.0%}",
              f"{c.top_country} {c.top_country_share:.0%}",
              c.top_country == country and c.top_country_share >= floor,
              "Residual country isn't being assigned, or country labels differ.")

    # 6. A gilt fund has no company or sector risk, and its 2022 fall was real.
    g = run["IGLT.L"]
    check("100% IGLT: no sector tilt", "0%", f"{g.concentration.top_sector_share:.0%}",
          g.concentration.top_sector_share == 0, "Bonds are being counted as a sector.")

    # 7. Prices: no leftover pence glitches (no ETF jumps more than 25% in a day).
    etfs = [t for t in universe["ticker"] if t in prices.columns]
    jumps = prices[etfs].pct_change(fill_method=None).abs().max()
    bad = jumps[jumps >= 0.25].sort_values(ascending=False)
    check("No ETF price jumps over 25% in a day", "< 25%",
          ", ".join(f"{t} {v:.0%}" for t, v in bad.items())
          or f"largest {jumps.idxmax()} {jumps.max():.0%}",
          bad.empty, "A data glitch that isn't a simple one-day spike: check that "
                     "ticker's prices around the date, or its currency in etf_universe.csv.")

    # 8. Every ETF price ends in a sensible pound range (catches pence left as pounds).
    last = prices[etfs].ffill().iloc[-1]
    odd = last[(last < 1) | (last > 1500)]
    check("ETF prices look like pounds (£1-£1,500)", "none outside",
          ", ".join(f"{t} £{v:,.0f}" for t, v in odd.items()) or "none", odd.empty,
          "Check the currency column for these tickers (GBp vs GBP vs USD).")

    # 9. The diversified demo must flag VAGP as estimated before it launched.
    est = [r.name for r in run["diversified"].scenarios if "VAGP.L" in r.estimated]
    check("Diversified demo: VAGP estimated pre-2019", "Brexit vote, Q4 2018 sell-off",
          ", ".join(est) or "none", {"Brexit vote", "Q4 2018 sell-off"} <= set(est),
          "Proxy for VAGP.L missing or misspelt in etf_universe.csv.")

    # 10. Every demo's company exposures add up to 100% of the money.
    sums = {d: run[d].companies["exposure"].sum() for d in demos}
    check("Demo exposures add to 100%", "100% each",
          ", ".join(f"{d} {v:.1%}" for d, v in sums.items()),
          all(abs(v - 1) < 0.001 for v in sums.values()), "Look-through is losing money.")

    table = pd.DataFrame(results)
    with pd.option_context("display.width", 220, "display.max_colwidth", 70):
        print("\n" + table[["result", "check", "expected", "actual"]].to_string(index=False))
    fails = table[table["result"] == "FAIL"]
    if not fails.empty:
        print(f"\n{len(fails)} surprise(s) to log and fix:")
        for row in fails.itertuples():
            print(f"  - {row.check}: {row.hint}")
        return 1
    print("\nAll sanity checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())