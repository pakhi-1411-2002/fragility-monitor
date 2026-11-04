"""Fragility Monitor: the Streamlit page.

Run from the repo root:
    streamlit run app/Home.py

Layout (plan section 9.1):
  1. Sidebar - choose a demo or upload a CSV; download the template.
  2. Verdict row - score dial + 3 headline cards.
  3. Top alerts - the three most serious findings, the rest folded away.
  4. Breakdown - which checks the score's points come from.
  5. Detail tabs - what you really own, fund overlap (more on Day 10).

Rule: this file only draws. Every number comes from fragility.pipeline.
"""

import logging

import components as ui
import streamlit as st

import data
from fragility.alerts import split_alerts
from fragility.config import DEMO_DIR, DISCLAIMER
from fragility.lookthrough import COMPANY, MIXED_COUNTRIES, UNCLASSIFIED
from fragility.portfolio import TEMPLATE_CSV

DEMOS = {
    "Overlapping (one US-tech bet)": "overlapping.csv",
    "Naive (leans UK and US)": "naive.csv",
    "Diversified (shares, bonds, gold)": "diversified.csv",
}

st.set_page_config(page_title="Fragility Monitor", page_icon=":material/shield:",
                   layout="wide")


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
            # Remember it, so switching to a demo and back doesn't lose the file.
            st.session_state["upload"] = (uploaded.name, uploaded.getvalue())
        if "upload" in st.session_state:
            label, csv_bytes = st.session_state["upload"]
            if uploaded is None:
                st.caption(f"Showing your last upload: **{label}**")
                if st.button("Forget this file", icon=":material/close:"):
                    del st.session_state["upload"]
                    st.rerun()

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
        report = data.analyse(csv_bytes)
    except Exception:  # noqa: BLE001 - never show a traceback to a user
        logging.getLogger("fragility.app").exception("Analysis failed for %s", label)
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
    with ui.guard("the score dial"):
        st.plotly_chart(ui.score_gauge(report.score.score, report.score.band),
                        width="stretch", config={"displayModeBar": False})
        ui.band_label(report.score.band, report.score.top_reason)

with c1:
    with ui.guard("the holdings card"):
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
    with ui.guard("the crash card"):
        worst = report.worst
        if worst is not None:
            ui.card("Worst past crash", "-" + ui.gbp(worst.worst_loss * report.total_value),
                    f"{worst.name} ({worst.start:%b %Y}): a {-worst.worst_loss:.0%} fall "
                    f"in your {ui.gbp(report.total_value)}.",
                    "What today's portfolio would have lost, held through that crash.")
        else:
            ui.card("Worst past crash", "n/a", "No price history to replay.")

with c3:
    with ui.guard("the biggest-company card"):
        companies = report.companies[report.companies["kind"] == COMPANY]
        if not companies.empty:
            top = companies.iloc[0]
            ui.card("Biggest single company", f"{top['exposure']:.1%}",
                    f"{top['company']}: {ui.gbp(top['exposure_gbp'])}, via {top['via']}.",
                    "Adds up every fund it hides in, plus any shares you hold directly.")
        else:
            ui.card("Biggest single company", "n/a", "No company-level holdings found.")

# ---------------------------------------------------------------------------
# 3. Top alerts: the three most serious findings
# ---------------------------------------------------------------------------
st.subheader("What stands out")
with ui.guard("the alerts"):
    shown, more = split_alerts(report.alerts)
    for column, alert in zip(st.columns(len(shown)), shown, strict=True):
        with column:
            ui.alert_card(alert)
    if more:
        with st.expander(f"More findings ({len(more)})"):
            for alert in more:
                ui.alert_card(alert)


# ---------------------------------------------------------------------------
# 4. Where the score comes from
# ---------------------------------------------------------------------------
st.subheader("Where your fragility comes from")
with ui.guard("the breakdown chart"):
    st.caption(f"Each check can add up to its share of the 100 points (grey). Coloured bars "
               f"show what it actually added; together they make your score of "
               f"{report.score.score:.0f}.")
    st.plotly_chart(ui.breakdown_bar(report.score.breakdown, report.score.band),
                    width="stretch", config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# 5. Detail tabs
# ---------------------------------------------------------------------------
st.subheader("The detail")
own_tab, overlap_tab = st.tabs(["What you really own", "Fund overlap"])

with own_tab:
    with ui.guard("the company and sector charts"):
        st.markdown("Every fund opened up and added together, plus any shares you hold "
                    "directly. Each bar shows **which of your holdings** that company "
                    "comes through.")
        st.plotly_chart(ui.companies_chart(report.companies, report.lines, report.total_value),
                        width="stretch", config={"displayModeBar": False})

        left, right = st.columns(2)
        with left:
            st.markdown("**By sector**")
            st.plotly_chart(ui.donut(report.sectors, "sector",
                                     {UNCLASSIFIED, "Unknown", "Bonds", "Commodities"}),
                            width="stretch", config={"displayModeBar": False})
        with right:
            st.markdown("**By country**")
            st.plotly_chart(ui.donut(report.countries, "country",
                                     {MIXED_COUNTRIES, "Unknown", "Global"}),
                            width="stretch", config={"displayModeBar": False})
        st.caption("Grey slices are funds' many smaller holdings (we keep each fund's top "
                   "50), plus bonds and gold, which have no company sector. So every "
                   "coloured slice is a minimum: the true share can only be higher.")

        with st.expander("Show as a table"):
            table = report.companies[["company", "exposure", "exposure_gbp", "via", "sector",
                                      "country"]]
            st.dataframe(table, hide_index=True, width="stretch", column_config={
                "company": "Company",
                "exposure": st.column_config.NumberColumn("Share", format="percent"),
                "exposure_gbp": st.column_config.NumberColumn("Amount", format="£%,.0f"),
                "via": "Comes through", "sector": "Sector", "country": "Country"})

with overlap_tab:
    with ui.guard("the fund overlap grid"):
        if report.overlap is None or len(report.overlap) < 2:
            st.info("You hold fewer than two funds, so there's nothing to compare.")
        else:
            st.markdown("How much each pair of your funds holds **in common**. Two funds at "
                        "90% are close to the same fund under two names: owning both adds "
                        "little variety.")
            st.plotly_chart(ui.overlap_heatmap(report.overlap), width="stretch",
                            config={"displayModeBar": False})
            st.caption("Counted company by company: for each company in both funds, take "
                       "the smaller of its two weights, then add them up. Only each fund's "
                       "top 50 holdings are compared, so the true overlap can be higher.")


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