"""Reusable pieces of the Streamlit page: charts and cards.

Everything here only DRAWS. All numbers come from the Report made by
fragility.pipeline; no maths happens in the app.
"""

import plotly.graph_objects as go
import streamlit as st

# Band colours: muted green / amber / red that read well in light and dark mode.
BAND_COLOURS = {"Robust": "#2e9d5b", "Moderate": "#e0a100", "Fragile": "#d64541",
                "Unknown": "#8a8f98"}
SEVERITY_ICONS = {3: ":material/error:", 2: ":material/warning:", 0: ":material/check_circle:"}


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