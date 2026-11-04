"""Cached data loading for the app (plan section 9.2).

Plain idea: like a kitchen doing its prep before service. The slow things
(reading the reference files, downloading prices, running the analysis) are
done once and kept on the side; the next request for the same thing is
served straight away.

  load_reference()      fund list, fund holdings, crash list  - kept 1 hour
  prices_for(tickers)   daily GBP prices for those tickers     - kept 6 hours
  analyse(csv_bytes)    the full Report for one portfolio CSV  - kept 6 hours

Streamlit recognises "the same thing" by the arguments: the same CSV bytes
or the same set of tickers give the stored answer without redoing the work.
"""

import io

import streamlit as st

from fragility.config import MARKET_TICKER
from fragility.pipeline import Report, run_analysis
from fragility.portfolio import load_portfolio
from fragility.prices import load_prices
from fragility.stress import load_scenarios, tickers_needed
from fragility.universe import load_holdings, load_universe, share_symbols


@st.cache_data(ttl="1h", show_spinner=False)
def load_reference() -> tuple:
    """(universe, holdings, scenarios, share_symbols) from data/reference/."""
    universe, holdings = load_universe(), load_holdings()
    return universe, holdings, load_scenarios(), share_symbols(holdings)


@st.cache_data(ttl="6h", show_spinner=False)
def prices_for(tickers: tuple):
    """Daily GBP prices for these tickers (a tuple, so Streamlit can remember it)."""
    return load_prices(list(tickers))


@st.cache_data(ttl="6h", show_spinner=False)
def analyse(csv_bytes: bytes) -> Report:
    """Check the CSV, load what it needs, and run the whole pipeline."""
    universe, holdings, scenarios, symbols = load_reference()
    checked = load_portfolio(io.BytesIO(csv_bytes), universe, share_symbols=symbols)
    if not checked.ok:
        return Report(portfolio=checked.portfolio, total_value=0.0,
                      errors=checked.errors, notices=checked.notices)

    needed = tickers_needed(checked.portfolio, universe) + [MARKET_TICKER]
    prices = prices_for(tuple(sorted(set(needed))))
    report = run_analysis(checked.portfolio, universe=universe, holdings=holdings,
                          prices=prices, scenarios=scenarios)
    report.errors, report.notices = checked.errors, checked.notices
    return report