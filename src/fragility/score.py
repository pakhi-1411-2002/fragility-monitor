"""The fragility score: one number from 0 (robust) to 100 (fragile), with its reasons.

Plain idea: like a car's MOT, four separate checks each get a mark out of
100, and the overall mark is a weighted average. Every point can be traced
back to one check, so the app can say WHY a portfolio scored what it did.

Each check turns a measurement into 0-100 with a straight line between a
"calm" value (0 points) and an "alarm" value (100 points), capped at both ends:

    sub_score = clip( 100 * (x - calm) / (alarm - calm), 0, 100 )

    Example, top-10 share: calm 15%, alarm 60%.
    40% -> 100 * (0.40 - 0.15) / (0.60 - 0.15) = 55.6 points

    score = sum over checks k of  weight_k * sub_score_k

All the numbers (calm, alarm, weight, bands) live in config.py.
"""

import math
from dataclasses import dataclass, field

import pandas as pd

from fragility.config import COUNTRY_SHARE_DIVISOR, SCORE_BANDS, SCORE_COMPONENTS

# What each check means, for the app and the alerts.
LABELS = {
    "company_concentration": "Too much money in a few companies",
    "sector_country_tilt": "Leaning heavily on one sector or country",
    "moves_together": "Holdings fall together on bad weeks",
    "crash_loss": "Big losses in past crashes",
}
METRICS = {
    "company_concentration": "Top-10 company share",
    "sector_country_tilt": f"Max of sector share and country share / {COUNTRY_SHARE_DIVISOR}",
    "moves_together": "Average correlation in bad market weeks",
    "crash_loss": "Worst historical crash loss",
}


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def scale(x: float, calm: float, alarm: float) -> float:
    """Straight line from calm (0) to alarm (100), capped at 0 and 100. NaN stays NaN."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float("nan")
    return float(min(100.0, max(0.0, 100 * (x - calm) / (alarm - calm))))


def tilt(sector_share: float, country_share: float,
         divisor: float = COUNTRY_SHARE_DIVISOR) -> float:
    """The bigger of the sector share and the (softened) country share.

    The country share is divided by 1.5 because some home or US bias is
    normal: a global tracker is already ~60% US, which shouldn't count as
    alarming on its own.
    """
    return max(sector_share or 0.0, (country_share or 0.0) / divisor)


def band(score: float, bands: list = SCORE_BANDS) -> str:
    """'Robust', 'Moderate' or 'Fragile' (score rounded to a whole number first)."""
    if math.isnan(score):
        return "Unknown"
    rounded = round(score)
    for upper, label in bands:
        if rounded <= upper:
            return label
    return bands[-1][1]


# ---------------------------------------------------------------------------
# The score
# ---------------------------------------------------------------------------
@dataclass
class ScoreResult:
    score: float
    band: str
    breakdown: pd.DataFrame = field(repr=False)

    @property
    def top_reason(self) -> str:
        """The check that added the most points."""
        if self.breakdown.empty or self.breakdown["points"].isna().all():
            return ""
        return self.breakdown.loc[self.breakdown["points"].idxmax(), "label"]


def fragility_score(metrics: dict, components: dict = SCORE_COMPONENTS) -> ScoreResult:
    """Score from raw measurements.

    metrics: {"company_concentration": 0.45, "sector_country_tilt": 0.60,
              "moves_together": 0.85, "crash_loss": 0.24}   (crash_loss as a positive fall)

    A check whose measurement is missing (NaN) is left out and the remaining
    weights are scaled up to add to 1, e.g. "moves together" can't be measured
    for a portfolio of one fund. The breakdown marks it as not available.
    """
    rows = []
    for key, cfg in components.items():
        value = metrics.get(key, float("nan"))
        rows.append({
            "component": key,
            "label": LABELS.get(key, key),
            "metric": METRICS.get(key, ""),
            "value": value,
            "sub_score": scale(value, cfg["calm"], cfg["alarm"]),
            "weight": cfg["weight"],
        })
    table = pd.DataFrame(rows)
    table["available"] = table["sub_score"].notna()

    total_weight = table.loc[table["available"], "weight"].sum()
    if total_weight == 0:
        table["used_weight"] = 0.0
        table["points"] = float("nan")
        return ScoreResult(float("nan"), "Unknown", table)

    table["used_weight"] = table["weight"].where(table["available"], 0.0) / total_weight
    table["points"] = table["used_weight"] * table["sub_score"].fillna(0.0)
    table.loc[~table["available"], "points"] = float("nan")

    score = float(table["points"].sum())
    return ScoreResult(score, band(score), table)


def metrics_from(concentration, correlation, worst_loss: float) -> dict:
    """Turn the module results into the four measurements the score needs.

    concentration: concentration.ConcentrationResult
    correlation:   correlation.CorrelationResult
    worst_loss:    stress.worst_scenario(results).worst_loss (negative, e.g. -0.24)
    """
    return {
        "company_concentration": concentration.top10_share,
        "sector_country_tilt": tilt(concentration.top_sector_share,
                                    concentration.top_country_share),
        "moves_together": correlation.avg_stress_corr,
        "crash_loss": -worst_loss if worst_loss is not None else float("nan"),
    }


def score_breakdown(result: ScoreResult) -> pd.DataFrame:
    """The 'where your fragility comes from' table for the app, biggest first."""
    cols = ["label", "value", "sub_score", "used_weight", "points", "available"]
    return result.breakdown[cols].sort_values("points", ascending=False,
                                              na_position="last").reset_index(drop=True)