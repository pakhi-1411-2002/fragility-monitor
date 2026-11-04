"""Alerts: simple if-then rules that turn a Report into plain-English warnings.

Plain idea: like the warning lights on a car dashboard. Each light watches
one thing and comes on past a set level. No AI writes these sentences: every
one comes from a rule below, so each can be traced to a number.

Rules (thresholds in config.py ALERT_THRESHOLDS):
  A1  One company is a big share, reached through 2+ routes
  A2  The top 10 companies hold a large share
  A3  Two funds hold largely the same companies
  A4  One sector is a large share
  A5  One country is a large share
  A6  Holdings fall together on bad weeks
  A7  A past crash would have cost a lot
  A8  Many holdings, but they move as very few groups
  P1  Positive: no major weak spots (shown when nothing else fires)

Severity:  3 = high (red), 2 = medium (amber), 0 = positive (green).
An alert is high when its number reaches 1.5x its threshold, or halfway
from the threshold to 100% if 1.5x is impossible (e.g. correlation 0.8 x 1.5).

Wording rule: state the fact and its consequence, never advice ("sell X").
"""

from dataclasses import dataclass, field

import pandas as pd

from fragility.config import (
    ALERT_THRESHOLDS,
    MAX_ALERTS_SHOWN,
    SEVERITY_HIGH_MULTIPLIER,
)
from fragility.lookthrough import COMPANY

HIGH, MEDIUM, POSITIVE = 3, 2, 0
SECTOR_SHOCK = -0.20     # used to put a £ figure on a sector tilt (A4)


@dataclass
class Alert:
    id: str
    severity: int
    title: str
    message: str
    impact_gbp: float = 0.0                     # used to sort alerts of equal severity
    evidence: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def high_level(threshold: float, cap: float | None = 1.0) -> float:
    """Where an alert turns from amber to red."""
    level = threshold * SEVERITY_HIGH_MULTIPLIER
    if cap is not None and level > cap:
        level = (threshold + cap) / 2
    return level


def severity(value: float, threshold: float, cap: float | None = 1.0) -> int:
    return HIGH if value >= high_level(threshold, cap) else MEDIUM


def gbp(x: float) -> str:
    return f"£{abs(x):,.0f}"


def pct(x: float) -> str:
    return f"{x:.0%}"


def _ok(x) -> bool:
    return x is not None and not pd.isna(x)


# ---------------------------------------------------------------------------
# The rules: each takes the Report and returns an Alert or None
# ---------------------------------------------------------------------------
def a1_hidden_company(r, t) -> Alert | None:
    companies = r.companies[r.companies["kind"] == COMPANY]
    hits = companies[(companies["exposure"] >= t["A1_single_company"])
                     & (companies["n_routes"] >= t["A1_min_routes"])]
    if hits.empty:
        return None
    top = hits.iloc[0]
    others = len(hits) - 1
    extra = f" {others} other compan{'y is' if others == 1 else 'ies are'} similar." \
        if others else ""
    return Alert(
        "A1", severity(top["exposure"], t["A1_single_company"], cap=None),
        "Hidden single-company bet",
        f"{top['company']} is {pct(top['exposure'])} of your money "
        f"({gbp(top['exposure_gbp'])}), through {int(top['n_routes'])} different "
        f"routes: {top['via']}.{extra}",
        impact_gbp=float(top["exposure_gbp"]),
        evidence={"companies": hits[["company", "exposure", "exposure_gbp", "via"]]
                  .to_dict("records")},
    )


def a2_top_heavy(r, t) -> Alert | None:
    share = r.concentration.top10_share
    if not _ok(share) or share < t["A2_top10_share"]:
        return None
    return Alert(
        "A2", severity(share, t["A2_top10_share"]), "Top-heavy",
        f"{pct(share)} of your money sits in just 10 companies "
        f"({gbp(share * r.total_value)}).",
        impact_gbp=share * r.total_value,
        evidence={"top10_share": share},
    )


def a3_duplicate_funds(r, t) -> Alert | None:
    m = r.overlap
    if m is None or m.empty or len(m) < 2:
        return None
    best, pair = 0.0, None
    funds = list(m.index)
    for i, a in enumerate(funds):
        for b in funds[i + 1:]:
            if m.loc[a, b] > best:
                best, pair = float(m.loc[a, b]), (a, b)
    if pair is None or best < t["A3_fund_overlap"]:
        return None
    value = r.portfolio.set_index("ticker")["value_gbp"]
    smaller = min(value.get(pair[0], 0), value.get(pair[1], 0))
    return Alert(
        "A3", severity(best, t["A3_fund_overlap"]), "Funds that overlap",
        f"{pair[0]} and {pair[1]} hold largely the same companies "
        f"({pct(best)} overlap), so owning both adds little variety.",
        impact_gbp=float(smaller),
        evidence={"pair": pair, "overlap": best},
    )


def a4_sector_tilt(r, t) -> Alert | None:
    c = r.concentration
    if not c.top_sector or c.top_sector_share < t["A4_sector_share"]:
        return None
    loss = c.top_sector_share * SECTOR_SHOCK * r.total_value
    return Alert(
        "A4", severity(c.top_sector_share, t["A4_sector_share"]), "Sector tilt",
        f"{pct(c.top_sector_share)} of your money is in {c.top_sector}. "
        f"A {pct(-SECTOR_SHOCK)} fall in that sector would cost about {gbp(loss)}.",
        impact_gbp=abs(loss),
        evidence={"sector": c.top_sector, "share": c.top_sector_share},
    )


def a5_country_tilt(r, t) -> Alert | None:
    c = r.concentration
    if not c.top_country or c.top_country_share < t["A5_country_share"]:
        return None
    return Alert(
        "A5", severity(c.top_country_share, t["A5_country_share"]), "Country tilt",
        f"{pct(c.top_country_share)} of your money depends on companies in "
        f"{c.top_country} ({gbp(c.top_country_share * r.total_value)}).",
        impact_gbp=c.top_country_share * r.total_value,
        evidence={"country": c.top_country, "share": c.top_country_share},
    )


def a6_moves_as_one(r, t) -> Alert | None:
    corr = r.correlation.avg_stress_corr if r.correlation else None
    if not _ok(corr) or corr < t["A6_stress_corr"]:
        return None
    return Alert(
        "A6", severity(corr, t["A6_stress_corr"]), "Falls together",
        "On bad market weeks, almost everything you own falls together "
        f"(average correlation {corr:.2f}, where 1.00 means perfectly in step).",
        impact_gbp=r.total_value * corr,
        evidence={"avg_stress_corr": corr},
    )


def a7_crash_loss(r, t) -> Alert | None:
    w = r.worst
    if w is None or not _ok(w.worst_loss) or -w.worst_loss < t["A7_crash_loss"]:
        return None
    loss = -w.worst_loss
    weeks = max(1, round((w.end - w.start).days / 7))
    return Alert(
        "A7", severity(loss, t["A7_crash_loss"]), "Big crash loss",
        f"In the {w.name} ({w.start:%b %Y}), this portfolio would have lost "
        f"{pct(loss)} ({gbp(loss * r.total_value)}) in about {weeks} weeks.",
        impact_gbp=loss * r.total_value,
        evidence={"scenario": w.name, "loss": loss},
    )


def a8_illusion_of_choice(r, t) -> Alert | None:
    n = len(r.portfolio)
    groups = r.correlation.effective_groups if r.correlation else None
    if n < t["A8_min_holdings"] or not _ok(groups) or groups > t["A8_max_effective_groups"]:
        return None
    sev = HIGH if groups <= t["A8_max_effective_groups"] * 0.75 else MEDIUM
    return Alert(
        "A8", sev, "Illusion of choice",
        f"You hold {n} investments, but they move like {groups:.1f} "
        f"group{'s' if round(groups, 1) != 1 else ''}.",
        impact_gbp=r.total_value / max(groups, 1),
        evidence={"holdings": n, "effective_groups": groups},
    )


RULES = [a1_hidden_company, a2_top_heavy, a3_duplicate_funds, a4_sector_tilt,
         a5_country_tilt, a6_moves_as_one, a7_crash_loss, a8_illusion_of_choice]


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------
def build_alerts(report, thresholds: dict = ALERT_THRESHOLDS) -> list:
    """Every alert that fires, most serious first (then biggest £ impact)."""
    alerts = [a for rule in RULES if (a := rule(report, thresholds)) is not None]
    alerts.sort(key=lambda a: (-a.severity, -a.impact_gbp))

    score = report.score.score if report.score is not None else float("nan")
    if not alerts or (_ok(score) and score <= thresholds["P1_max_score"]):
        alerts.append(Alert("P1", POSITIVE, "No major weak spots",
                            "No major weak spots found: your money is spread across "
                            "investments that don't all depend on the same thing."))
    return alerts


def split_alerts(alerts: list, n: int = MAX_ALERTS_SHOWN) -> tuple:
    """(top n to show on the main page, the rest for 'More findings')."""
    return alerts[:n], alerts[n:]