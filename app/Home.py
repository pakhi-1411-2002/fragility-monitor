"""Fragility Monitor: the Streamlit page.

Run from the repo root:
    streamlit run app/Home.py

Layout (plan section 9.1):
  1. Sidebar - choose a demo or upload a CSV; download the template.
  2. Verdict row - score dial + 3 headline cards.
  (Alerts, breakdown and detail tabs come next, on Days 8-10.)

Rule: this file only draws. Every number comes from fragility.pipeline.
"""

import io

import components as ui
import streamlit as st

from fragility.config import DEMO_DIR, DISCLAIMER
from fragility.lookthrough import COMPANY
from fragility.pipeline import analyse_csv
from fragility.portfolio import TEMPLATE_CSV

DEMOS = {
    "Overlapping (one US-tech bet)": "overlapping.csv",
    "Naive (leans UK and US)": "naive.csv",
    "Diversified (shares, bonds, gold)": "diversified.csv",
}

st.set_page_config(page_title="Fragility Monitor", page_icon=":material/shield:",
                   layout="wide")


@st.cache_data(show_spinner=False)
def analyse(csv_bytes: bytes):
    """Run the whole pipeline once per distinct CSV, then reuse the result."""
    return analyse_csv(io.BytesIO(csv_bytes))


# ---------------------------------------------------------------------------
# 1. Sidebar: where the portfolio comes from
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Your portfolio")
    source = st.radio("Start from", ["Try a demo", "Upload my CSV"], horizontal=True,
                      label_visibility="collapsed")

    csv_bytes, label = None, ""
    if source == "Try a demo":
        choice = st.selectbox("Demo portfolio (£25,000 each)", list(DEMOS))
        csv_bytes = (DEMO_DIR / DEMOS[choice]).read_bytes()
        label = choice.split(" (")[0] + " demo"
    else:
        uploaded = st.file_uploader("Portfolio CSV", type=["csv"],
                                    help="Two columns: ticker and value_gbp, "
                                         "one row per holding.")
        if uploaded is not None:
            csv_bytes, label = uploaded.getvalue(), uploaded.name

    st.download_button("Download CSV template", TEMPLATE_CSV, file_name="portfolio.csv",
                       mime="text/csv", icon=":material/download:", width="stretch")
    st.caption("Tickers as on the London Stock Exchange (e.g. VWRL.L) or US shares "
               "(e.g. AAPL). Values in pounds.")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Fragility Monitor")
st.markdown("**Is your portfolio as diversified as you think?** See what you really own, "
            "what falls together, and what past crashes would have cost.")

if csv_bytes is None:
    st.info("Upload a CSV in the sidebar, or switch to **Try a demo** to see an example.",
            icon=":material/upload_file:")
    st.stop()

with st.spinner(f"Analysing {label}..."):
    try:
        report = analyse(csv_bytes)
    except Exception:  # noqa: BLE001 - never show a traceback to a user
        st.error("Something went wrong while analysing this portfolio. Check the file "
                 "matches the template, or try again in a minute if price data is "
                 "temporarily unavailable.", icon=":material/error:")
        st.stop()

ui.message_list("warning", "Some rows couldn't be used", report.errors)
ui.message_list("info", "We tidied a few things", report.notices)

if not report.ok:
    st.warning("There's nothing left to analyse. Fix the rows above and upload again.")
    st.stop()


# ---------------------------------------------------------------------------
# 2. Verdict row: dial + three headline cards
# ---------------------------------------------------------------------------
st.subheader(f"Verdict: {label}")
dial, c1, c2, c3 = st.columns([1.3, 1, 1, 1], vertical_alignment="center")

with dial:
    st.plotly_chart(ui.score_gauge(report.score.score, report.score.band),
                    width="stretch", config={"displayModeBar": False})
    ui.band_label(report.score.band, report.score.top_reason)

with c1:
    n_lines = len(report.portfolio)
    groups = report.correlation.effective_groups if report.correlation else float("nan")
    if groups == groups and n_lines > 1:            # not NaN
        ui.card("Holdings vs real bets", f"{n_lines} → {groups:.1f}",
                f"You hold {n_lines} investments; they move like {groups:.1f} "
                f"separate group{'s' if round(groups, 1) != 1 else ''}.",
                "Holdings that rise and fall together count as one group.")
    else:
        bets = report.concentration.effective_n
        ui.card("Separate bets inside", f"{bets:.0f}",
                f"Your {n_lines} holding{'s' if n_lines != 1 else ''} spread over about "
                f"{bets:.0f} equal-sized company bets.")

with c2:
    worst = report.worst
    if worst is not None:
        ui.card("Worst past crash", "-" + ui.gbp(worst.worst_loss * report.total_value),
                f"{worst.name} ({worst.start:%b %Y}): a {-worst.worst_loss:.0%} fall "
                f"in your {ui.gbp(report.total_value)}.",
                "What today's portfolio would have lost, held through that crash.")
    else:
        ui.card("Worst past crash", "n/a", "No price history to replay.")

with c3:
    companies = report.companies[report.companies["kind"] == COMPANY]
    if not companies.empty:
        top = companies.iloc[0]
        ui.card("Biggest single company", f"{top['exposure']:.1%}",
                f"{top['company']}: {ui.gbp(top['exposure_gbp'])}, via {top['via']}.",
                "Adds up every fund it hides in, plus any shares you hold directly.")
    else:
        ui.card("Biggest single company", "n/a", "No company-level holdings found.")

if report.warnings:
    with st.expander(f"Notes about these results ({len(report.warnings)})"):
        for w in report.warnings:
            st.markdown(f"- {w}")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
dates = []
if report.holdings_as_of:
    dates.append(f"fund holdings as of {report.holdings_as_of}")
if report.prices_as_of:
    dates.append(f"prices to {report.prices_as_of}")
st.caption(f"{DISCLAIMER} " + ("Data: " + ", ".join(dates) + "." if dates else ""))