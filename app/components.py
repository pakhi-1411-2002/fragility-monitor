"""Reusable pieces of the Streamlit page: charts and cards.

Everything here only DRAWS. All numbers come from the Report made by
fragility.pipeline; no maths happens in the app.
"""

import logging
from contextlib import contextmanager

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

log = logging.getLogger("fragility.app")

# Band colours: muted green / amber / red that read well in light and dark mode.
BAND_COLOURS = {"Robust": "#2e9d5b", "Moderate": "#e0a100", "Fragile": "#d64541",
                "Unknown": "#8a8f98"}
SEVERITY_ICONS = {3: ":material/error:", 2: ":material/warning:", 0: ":material/check_circle:"}


@contextmanager
def guard(what: str):
    """Draw one part of the page; if it breaks, say so politely and carry on.

    The full error still goes to the terminal (log.exception), so you can fix
    it, but a visitor never sees a Python traceback. st.stop() and reruns are
    not Exceptions in Streamlit, so they pass straight through.
    """
    try:
        yield
    except Exception:  # noqa: BLE001 - one broken chart must not break the page
        log.exception("Failed to draw %s", what)
        st.warning(f"Couldn't show {what} right now. The rest of the page is unaffected.",
                   icon=":material/warning:")


def gbp(x: float) -> str:
    """£12,345 (always positive: say 'lost' or 'cost' in the sentence instead)."""
    return f"£{abs(x):,.0f}"


def score_gauge(score: float, band: str) -> go.Figure:
    """Semi-circle dial from 0 to 100 with Robust / Moderate / Fragile zones."""
    colour = BAND_COLOURS.get(band, BAND_COLOURS["Unknown"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score),
        number={"suffix": " / 100", "font": {"size": 34}},
        gauge={
            "axis": {"range": [0, 100], "tickvals": [0, 33, 66, 100]},
            "bar": {"color": colour, "thickness": 0.3},
            "steps": [
                {"range": [0, 33], "color": "rgba(46,157,91,0.15)"},
                {"range": [33, 66], "color": "rgba(224,161,0,0.15)"},
                {"range": [66, 100], "color": "rgba(214,69,65,0.15)"},
            ],
        },
    ))
    fig.update_layout(height=210, margin={"t": 20, "b": 0, "l": 30, "r": 30},
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def band_label(band: str, reason: str) -> None:
    """The verdict in words under the dial, e.g. 'Fragile' + its main reason."""
    colour = BAND_COLOURS.get(band, BAND_COLOURS["Unknown"])
    st.markdown(f"<div style='text-align:center;font-size:1.4rem;font-weight:700;"
                f"color:{colour}'>{band}</div>", unsafe_allow_html=True)
    if reason:
        st.markdown(f"<div style='text-align:center;opacity:0.8'>Main reason: "
                    f"{reason.lower()}</div>", unsafe_allow_html=True)


def card(title: str, value: str, caption: str = "", help_text: str = "") -> None:
    """A bordered box with a small title, a big number and one line under it."""
    with st.container(border=True):
        st.metric(title, value, help=help_text or None)
        if caption:
            st.caption(caption)


def message_list(kind: str, title: str, lines: list) -> None:
    """Show a list of messages as one Streamlit box (kind: 'warning' or 'info')."""
    if not lines:
        return
    text = f"**{title}**\n\n" + "\n".join(f"- {line}" for line in lines)
    getattr(st, kind)(text)


# ---------------------------------------------------------------------------
# Alert cards (plan 9.1 item 3)
# ---------------------------------------------------------------------------
SEVERITY_BOX = {3: "error", 2: "warning", 0: "success"}     # red, amber, green


def _evidence(alert) -> None:
    """The numbers behind an alert, in a readable form."""
    ev = alert.evidence or {}
    if "companies" in ev:
        table = pd.DataFrame(ev["companies"]).rename(columns={
            "company": "Company", "exposure": "Share of your money",
            "exposure_gbp": "Amount", "via": "Reached through"})
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "Share of your money": st.column_config.NumberColumn(format="percent"),
            "Amount": st.column_config.NumberColumn(format="£%,.0f")})
        return
    for key, value in ev.items():
        name = key.replace("_", " ").capitalize()
        if isinstance(value, float):
            shown = f"{value:.2f}" if "corr" in key or "groups" in key else f"{value:.1%}"
        elif isinstance(value, tuple):
            shown = " and ".join(value)
        else:
            shown = str(value)
        st.markdown(f"- {name}: **{shown}**")


def alert_card(alert) -> None:
    """One coloured box: title + plain-English message, numbers in a fold-out."""
    box = getattr(st, SEVERITY_BOX.get(alert.severity, "info"))
    box(f"**{alert.title}**\n\n{alert.message}", icon=SEVERITY_ICONS.get(alert.severity))
    if alert.evidence:
        with st.expander("Show the numbers"):
            _evidence(alert)


# ---------------------------------------------------------------------------
# Fragility breakdown bar (plan 9.1 item 4)
# ---------------------------------------------------------------------------
def _measurement(component: str, value: float) -> str:
    if value != value:                     # NaN
        return "not measured"
    if component == "moves_together":
        return f"correlation {value:.2f}"
    return f"{value:.0%}"


def breakdown_bar(breakdown, band: str) -> go.Figure:
    """Horizontal bars: points each check adds to the score, out of its maximum.

    The faint bar behind is the most that check could add (its weight x 100),
    so the length of the solid bar reads as "how much of its share it used".
    The solid bars add up exactly to the score.
    """
    b = breakdown.copy()
    b["max_points"] = b["used_weight"] * 100
    b["points"] = b["points"].fillna(0.0)
    b = b.sort_values("points")            # biggest ends up on top
    labels = [lab if ok else f"{lab} (not measured)"
              for lab, ok in zip(b["label"], b["available"], strict=True)]
    hover = [f"{_measurement(c, v)} → {p:.1f} of {m:.0f} points"
             for c, v, p, m in zip(b["component"], b["value"], b["points"], b["max_points"],
                                   strict=True)]
    colour = BAND_COLOURS.get(band, BAND_COLOURS["Unknown"])

    fig = go.Figure()
    fig.add_bar(y=labels, x=b["max_points"], orientation="h", hoverinfo="skip",
                marker={"color": "rgba(138,143,152,0.18)"}, showlegend=False)
    fig.add_bar(y=labels, x=b["points"], orientation="h", marker={"color": colour},
                text=[f"{p:.0f} pts" for p in b["points"]], textposition="outside",
                hovertext=hover, hoverinfo="text", showlegend=False)
    fig.update_layout(
        barmode="overlay", height=60 + 52 * len(b), bargap=0.35,
        margin={"t": 10, "b": 10, "l": 10, "r": 40},
        xaxis={"range": [0, max(40, float(b["max_points"].max()) * 1.15)],
               "title": "points added to the score", "gridcolor": "rgba(138,143,152,0.2)"},
        yaxis={"automargin": True},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# Tab: What you really own (plan 9.1 item 5)
# ---------------------------------------------------------------------------
ROUTE_COLOURS = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#72b7b2", "#eeca3b",
                 "#9d755d", "#ff9da6"]
DIRECT_COLOUR = "#3b3f45"
GREY = "#b8bcc4"


def companies_chart(companies: pd.DataFrame, lines: pd.DataFrame, total: float,
                    n: int = 15) -> go.Figure:
    """Top n companies you really own, each bar split by the route it comes through."""
    top = companies[companies["kind"] == "company"].head(n)
    routes = (lines[lines["holding_id"].isin(top["holding_id"])]
              .groupby(["holding_id", "source"], as_index=False)["exposure"].sum())
    order = list(top["holding_id"])[::-1]                 # biggest at the top
    names = dict(zip(top["holding_id"], top["company"], strict=True))
    sources = sorted(set(routes["source"]) - {"direct"})
    sources += ["direct"] if "direct" in set(routes["source"]) else []

    fig = go.Figure()
    for i, source in enumerate(sources):
        part = routes[routes["source"] == source].set_index("holding_id").reindex(order)
        values = part["exposure"].fillna(0.0)
        fig.add_bar(
            y=[names[h] for h in order], x=values, orientation="h",
            name="Held directly" if source == "direct" else source,
            marker={"color": DIRECT_COLOUR if source == "direct"
                    else ROUTE_COLOURS[i % len(ROUTE_COLOURS)]},
            customdata=values * total,
            hovertemplate="%{y}<br>" + ("held directly" if source == "direct"
                                         else f"via {source}")
                          + ": %{x:.1%} (£%{customdata:,.0f})<extra></extra>",
        )
    fig.update_layout(
        barmode="stack", height=90 + 26 * len(order),
        margin={"t": 10, "b": 10, "l": 10, "r": 10},
        xaxis={"tickformat": ".0%", "title": "share of your money",
               "gridcolor": "rgba(138,143,152,0.2)"},
        yaxis={"automargin": True},
        legend={"orientation": "h", "y": -0.12, "title": {"text": "Comes through: "}},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def donut(table: pd.DataFrame, label_col: str, grey_labels: set, n: int = 7) -> go.Figure:
    """Share by sector or country: the n biggest named slices, then 'Other'.

    Slices that aren't a real sector/country (funds' unclassified small
    holdings, bonds, gold) are drawn grey so the eye skips them.
    """
    t = table[[label_col, "exposure"]].copy()
    named = t[~t[label_col].isin(grey_labels)].sort_values("exposure", ascending=False)
    greys = t[t[label_col].isin(grey_labels)]
    head, tail = named.head(n), named.iloc[n:]
    if not tail.empty:
        head = pd.concat([head, pd.DataFrame({label_col: ["Other"],
                                              "exposure": [tail["exposure"].sum()]})])
    data = pd.concat([head, greys])
    colours = [GREY if lab in grey_labels else ROUTE_COLOURS[i % len(ROUTE_COLOURS)]
               for i, lab in enumerate(data[label_col])]
    total = data["exposure"].sum()
    text = [f"{v / total:.0%}" if total and v / total >= 0.04 else ""   # hide tiny labels
            for v in data["exposure"]]
    fig = go.Figure(go.Pie(labels=data[label_col], values=data["exposure"], hole=0.55,
                           sort=False, marker={"colors": colours}, text=text,
                           textinfo="text", textposition="inside",
                           hovertemplate="%{label}: %{percent}<extra></extra>"))
    fig.update_layout(height=340, margin={"t": 10, "b": 10, "l": 10, "r": 10},
                      legend={"font": {"size": 11}}, paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ---------------------------------------------------------------------------
# Tab: Fund overlap (plan 9.1 item 5)
# ---------------------------------------------------------------------------
def overlap_heatmap(matrix: pd.DataFrame) -> go.Figure:
    """Grid of how much each pair of your funds holds in common (0-100%)."""
    funds = list(matrix.index)
    values = matrix.to_numpy(dtype=float)
    text = [["" if a == b else f"{values[i][j]:.0%}" for j, b in enumerate(funds)]
            for i, a in enumerate(funds)]
    shown = values.copy()
    for i in range(len(funds)):
        shown[i][i] = float("nan")                         # a fund vs itself: blank
    fig = go.Figure(go.Heatmap(
        z=shown, x=funds, y=funds, text=text, texttemplate="%{text}", zmin=0, zmax=1,
        colorscale=[[0, "#f4f5f7"], [0.5, "#f2b24e"], [1, "#d64541"]],
        hovertemplate="%{y} and %{x}: %{z:.0%} in common<extra></extra>",
        colorbar={"tickformat": ".0%", "title": "in common"},
    ))
    fig.update_layout(height=120 + 55 * len(funds),
                      margin={"t": 10, "b": 10, "l": 10, "r": 10},
                      xaxis={"showgrid": False},
                      yaxis={"autorange": "reversed", "showgrid": False},
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ---------------------------------------------------------------------------
# Tab: Moves together (plan 9.1 item 5)
# ---------------------------------------------------------------------------
def corr_heatmap(corr: pd.DataFrame, order: list) -> go.Figure:
    """How closely each pair of holdings moved: -1 opposite, 0 unrelated, +1 in step.

    `order` puts holdings that belong to the same group next to each other,
    so groups show up as warm blocks along the diagonal.
    """
    tickers = [t for t in order if t in corr.index]
    c = corr.reindex(index=tickers, columns=tickers)
    values = c.to_numpy(dtype=float, copy=True)
    text = [["" if i == j or v != v else f"{v:.2f}" for j, v in enumerate(row)]
            for i, row in enumerate(values)]
    for i in range(len(tickers)):
        values[i][i] = float("nan")                        # a holding vs itself: blank
    fig = go.Figure(go.Heatmap(
        z=values, x=tickers, y=tickers, text=text, texttemplate="%{text}",
        zmin=-1, zmax=1,
        colorscale=[[0, "#4c78a8"], [0.5, "#f4f5f7"], [1, "#d64541"]],
        hovertemplate="%{y} and %{x}: %{z:.2f}<extra></extra>",
        colorbar={"title": "moves<br>together", "tickvals": [-1, 0, 1],
                  "ticktext": ["opposite", "unrelated", "in step"]},
    ))
    fig.update_layout(height=120 + 55 * len(tickers),
                      margin={"t": 10, "b": 10, "l": 10, "r": 10},
                      xaxis={"showgrid": False},
                      yaxis={"autorange": "reversed", "showgrid": False},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ---------------------------------------------------------------------------
# Tab: Crash replay (plan 9.1 item 5)
# ---------------------------------------------------------------------------
def crash_path_chart(result, total: float) -> go.Figure:
    """Value of today's portfolio, day by day, held through one past crash."""
    value = result.path * total
    low_day = value.idxmin()
    fig = go.Figure()
    fig.add_scatter(x=value.index, y=value, mode="lines", line={"color": "#4c78a8",
                                                                 "width": 2.5},
                    hovertemplate="%{x|%d %b %Y}: £%{y:,.0f}<extra></extra>",
                    showlegend=False)
    fig.add_hline(y=total, line={"dash": "dot", "color": GREY},
                  annotation_text=f"start: {gbp(total)}", annotation_position="top left")
    fig.add_scatter(x=[low_day], y=[value[low_day]], mode="markers+text",
                    marker={"color": "#d64541", "size": 10},
                    text=[f"lowest: {gbp(value[low_day])}"], textposition="bottom center",
                    hoverinfo="skip", showlegend=False)
    fig.update_layout(height=340, margin={"t": 20, "b": 10, "l": 10, "r": 10},
                      yaxis={"tickprefix": "£", "tickformat": ",.0f",
                             "gridcolor": "rgba(138,143,152,0.2)"},
                      xaxis={"showgrid": False},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig