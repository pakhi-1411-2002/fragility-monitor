"""Fragility Monitor: the Streamlit page.

Run from the repo root:
    streamlit run app/Home.py

Layout (plan section 9.1):
  1. Sidebar - choose a demo, upload a CSV or type holdings in; download the template.
  2. Verdict row - score dial + 3 headline cards.
  3. Top alerts - the three most serious findings, the rest folded away.
  4. Breakdown - which checks the score's points come from.
  5. Detail tabs - what you really own, fund overlap, moves together,
     crash replay, method.

Rule: this file only draws. Every number comes from fragility.pipeline.
"""

import logging
import re

import components as ui
import pandas as pd
import streamlit as st

import data
from fragility.alerts import split_alerts
from fragility.config import DEMO_DIR, DISCLAIMER, SCORE_COMPONENTS, TOP_N_HOLDINGS
from fragility.lookthrough import COMPANY, MIXED_COUNTRIES, UNCLASSIFIED
from fragility.portfolio import TEMPLATE_CSV
from fragility.score import LABELS, METRICS

METHOD_TEXT = """
**What this tool does, in plain words**

1. **Look-through.** Every fund is opened up, like emptying several boxes of chocolates
   onto one table, and the same company is added up across all of them, plus any shares
   you hold directly.
2. **Concentration.** How much of your money sits in a few companies, sectors or
   countries. Squaring each share and adding them up (the HHI) gives the "effective
   number of bets": ten equal holdings = 10, one holding = 1.
3. **Moves together.** Week-by-week correlation between your holdings, mainly on weeks
   when the global market fell more than 2%, because that's when spreading out matters.
4. **Crash replay.** Today's portfolio, held unchanged through real past crashes, using
   each holding's actual prices in pounds.
5. **Score.** Four checks each give 0-100 points on a straight line between a "calm" and
   an "alarm" level; the score is their weighted average, so every point can be traced
   to one check.
"""

LIMITS_TEXT = """
**Limits to keep in mind**

- Only each fund's **top {TOP_N_HOLDINGS} holdings** are named; the rest is treated as many small
  companies. So company, sector and overlap figures are **minimums**.
- Holdings are a **snapshot** from the date above; funds change slowly over time.
- **History is not a forecast.** A portfolio that did well in past crashes can still do
  badly in the next one.
- Bond and gold funds are treated as **one block each**, not looked through.
- **This is an educational tool, not financial advice.** It describes risk; it doesn't
  tell you what to buy or sell.
"""


def _fmt_threshold(key: str, value: float) -> str:
    return f"{value:.2f}" if key == "moves_together" else f"{value:.0%}"

DEMOS = {
    "Overlapping (one US-tech bet)": "overlapping.csv",
    "Naive (leans UK and US)": "naive.csv",
    "Diversified (shares, bonds, gold)": "diversified.csv",
}

st.set_page_config(page_title="Fragility Monitor", page_icon=":material/shield:",
                   layout="wide")


# ---------------------------------------------------------------------------
# Manual entry (plan 9.1 item 1)
# ---------------------------------------------------------------------------
STARTER_ROWS = pd.DataFrame({"ticker": ["VWRL.L", "EQQQ.L", "AAPL"],
                             "value_gbp": [12000.0, 5000.0, 2500.0]})


def typed_portfolio() -> tuple:
    """An editable table in the sidebar. Returns (csv_bytes, label) once submitted.

    It sits inside a form, so typing doesn't re-run the analysis on every
    keystroke: nothing happens until "Analyse these holdings" is pressed.
    The table goes through exactly the same checks as an uploaded CSV.
    """
    universe = data.load_reference()[0]
    with st.form("typed", border=False):
        edited = st.data_editor(
            st.session_state.get("typed_rows", STARTER_ROWS),
            num_rows="dynamic", hide_index=True, width="stretch",
            column_config={
                "ticker": st.column_config.TextColumn(
                    "Ticker", help="e.g. VWRL.L for a London fund, AAPL for a US share",
                    required=True),
                "value_gbp": st.column_config.NumberColumn(
                    "Value (£)", min_value=0.0, step=100.0, format="£%,.0f",
                    required=True),
            })
        submitted = st.form_submit_button("Analyse these holdings", type="primary",
                                          icon=":material/play_arrow:", width="stretch")
    if submitted:
        rows = edited.dropna(how="all")
        st.session_state["typed_rows"] = rows.reset_index(drop=True)
        st.session_state["typed_csv"] = rows.to_csv(index=False).encode()
    with st.expander("Funds we can look inside"):
        st.dataframe(universe[["ticker", "name"]], hide_index=True, width="stretch",
                     column_config={"ticker": "Ticker", "name": "Fund"})
    if "typed_csv" in st.session_state:
        return st.session_state["typed_csv"], "your holdings"
    return None, ""


# ---------------------------------------------------------------------------
# 1. Sidebar: where the portfolio comes from
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Your portfolio")
    source = st.radio("Start from", ["Try a demo", "Upload a CSV", "Type it in"],
                      label_visibility="collapsed")

    csv_bytes, label = None, ""
    if source == "Try a demo":
        choice = st.selectbox("Demo portfolio (£25,000 each)", list(DEMOS))
        csv_bytes = (DEMO_DIR / DEMOS[choice]).read_bytes()
        label = choice.split(" (")[0] + " demo"
    elif source == "Type it in":
        csv_bytes, label = typed_portfolio()
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
    if source == "Type it in":
        st.info("Add your holdings in the sidebar table, then press **Analyse these "
                "holdings**.", icon=":material/edit_note:")
    else:
        st.info("Upload a CSV in the sidebar, or switch to **Try a demo** to see an "
                "example.", icon=":material/upload_file:")
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

errors = report.errors
if source == "Type it in":
    # Row numbers count the CSV header as row 1; in the typed table, row 1 is the first
    # holding, so shift them down by one.
    errors = [re.sub(r"^Row (\d+)", lambda m: f"Row {int(m.group(1)) - 1}", e)
              for e in errors]
ui.message_list("warning", "Some rows couldn't be used", errors)
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
own_tab, overlap_tab, moves_tab, crash_tab, method_tab = st.tabs(
    ["What you really own", "Fund overlap", "Moves together", "Crash replay", "Method"])

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
                   f"{TOP_N_HOLDINGS}), plus bonds and gold, which have no company "
                   "sector. So every coloured slice is a minimum: the true share can "
                   "only be higher.")

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
                       f"top {TOP_N_HOLDINGS} holdings are compared, so the true overlap "
                       "can be higher.")


with moves_tab, ui.guard("the moves-together view"):
    corr = report.correlation
    if corr is None or len(report.portfolio) < 2 or corr.corr.empty:
        st.info("You hold one investment, so there's nothing to compare it with.")
    else:
        st.markdown("Diversification only protects you if your holdings **don't all fall "
                    "on the same days**. This compares how they moved week by week.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Bad weeks: average", f"{corr.avg_stress_corr:.2f}",
                  help=f"Only the {corr.n_stress_weeks} weeks since 2015 when the global "
                       "market fell more than 2%. 1.00 = perfectly in step.")
        m2.metric("Normal weeks: average", f"{corr.avg_corr:.2f}",
                  help="All weeks in the last 3 years.")
        m3.metric("Diversification ratio", f"{corr.diversification_ratio:.2f}",
                  help="1.00 = mixing your holdings doesn't calm the ride at all; "
                       "1.40 = the mix is about 30% calmer than its parts.")

        view = st.radio("Which weeks", ["Bad market weeks", "All weeks (last 3 years)"],
                        horizontal=True)
        matrix = corr.stress_corr if view.startswith("Bad") else corr.corr
        order = [t for group in corr.groups for t in group]
        st.plotly_chart(ui.corr_heatmap(matrix, order), width="stretch",
                        config={"displayModeBar": False})

        st.markdown(f"**Your holdings move like {corr.effective_groups:.1f} separate "
                    f"group{'s' if round(corr.effective_groups, 1) != 1 else ''}:**")
        weights = report.portfolio.set_index("ticker")["weight"]
        for i, group in enumerate(corr.groups, start=1):
            share = weights.reindex(group).fillna(0).sum()
            together = "these move together" if len(group) > 1 else "moves on its own"
            st.markdown(f"{i}. **{share:.0%} of your money**: {', '.join(group)} "
                        f"({together})")

with crash_tab, ui.guard("the crash replay"):
    st.markdown("Today's portfolio, held unchanged through past crashes. Each holding "
                "moves exactly as it really did, in pounds.")
    table = report.scenario_table.copy()
    if table.empty:
        st.info("No price history to replay.")
    else:
        shown_table = {
            "Crash": table["scenario"],
            "Dates": table["dates"],
            "Start to end": table["return"].map(lambda v: f"{v:+.1%}"),
            "In pounds": table["loss_gbp"].map(
                lambda v: f"{'-' if v < 0 else '+'}{ui.gbp(v)}"),
            "Worst dip": table["max_drawdown"].map(lambda v: f"{v:.1%}"),
            "Hit hardest": table["worst_contributors"],
            "Estimated?": table["estimated"].map(lambda t: f"yes ({t})" if t else ""),
        }
        st.dataframe(shown_table, hide_index=True, width="stretch")
        st.caption("'Worst dip' is the deepest fall at any point during the crash, "
                   "which can be worse than start-to-end. 'Estimated' means a fund "
                   "didn't exist yet and a similar older fund stood in for it.")

        valid = [r for r in report.scenarios if not r.path.empty]
        names = [r.name for r in valid]
        worst_name = report.worst.name if report.worst is not None else ""
        default = names.index(worst_name) if worst_name in names else 0
        picked = st.selectbox("See it day by day", names, index=default)
        result = valid[names.index(picked)]
        st.plotly_chart(ui.crash_path_chart(result, report.total_value), width="stretch",
                        config={"displayModeBar": False})

    st.markdown("**What if, tomorrow?** Made-up shocks, applied to what you really own:")
    for row in report.hypotheticals.itertuples():
        if row.exposed < 0.005:
            st.markdown(f"- **{row.scenario}**: almost none of your money is exposed.")
        else:
            st.markdown(f"- **{row.scenario}**: {row.exposed:.0%} of your money is "
                        f"exposed, so it would cost about **{ui.gbp(row.loss_gbp)}**.")

with method_tab:
    st.markdown(METHOD_TEXT)
    st.markdown("**How the score is built**")
    st.dataframe(
        [{"Check": LABELS[k], "Measured by": METRICS[k],
          "0 points at": _fmt_threshold(k, c["calm"]),
          "100 points at": _fmt_threshold(k, c["alarm"]), "Weight": f"{c['weight']:.0%}"}
         for k, c in SCORE_COMPONENTS.items()],
        hide_index=True, width="stretch")
    st.caption("Bands: 0-33 Robust, 34-66 Moderate, 67-100 Fragile. All settings live in "
               "config.py and were checked against five reference portfolios.")
    st.markdown("**Data used for this result**")
    st.markdown(f"- Fund holdings as of **{report.holdings_as_of or 'unknown'}** (from each "
                "fund provider's published list)\n"
                f"- Daily prices up to **{report.prices_as_of or 'unknown'}** (Yahoo "
                "Finance, converted to pounds)\n"
                "- Crashes replayed: "
                + "; ".join(f"{r.name} ({r.start:%d %b %Y} to {r.end:%d %b %Y})"
                            for r in report.scenarios))
    st.markdown(LIMITS_TEXT.format(TOP_N_HOLDINGS=TOP_N_HOLDINGS))


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