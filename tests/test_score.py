"""Tests for fragility.score, with sums worked out by hand."""

import math
from types import SimpleNamespace

import pytest

from fragility.config import SCORE_COMPONENTS
from fragility.score import (
    band,
    fragility_score,
    metrics_from,
    scale,
    score_breakdown,
    tilt,
)

ALL_CALM = {k: c["calm"] for k, c in SCORE_COMPONENTS.items()}
ALL_ALARM = {k: c["alarm"] for k, c in SCORE_COMPONENTS.items()}


# --- the straight line ------------------------------------------------------------
def test_scale_calm_is_0_alarm_is_100_middle_is_50():
    assert scale(0.15, 0.15, 0.60) == 0
    assert scale(0.60, 0.15, 0.60) == 100
    assert scale(0.375, 0.15, 0.60) == pytest.approx(50)


def test_scale_is_capped_both_ends():
    assert scale(0.05, 0.15, 0.60) == 0
    assert scale(0.95, 0.15, 0.60) == 100


def test_scale_keeps_missing_as_missing():
    assert math.isnan(scale(float("nan"), 0.15, 0.60))


# --- settings sanity ---------------------------------------------------------------
def test_weights_add_up_to_one():
    assert sum(c["weight"] for c in SCORE_COMPONENTS.values()) == pytest.approx(1.0)


def test_calm_is_below_alarm_everywhere():
    assert all(c["calm"] < c["alarm"] for c in SCORE_COMPONENTS.values())


# --- tilt ---------------------------------------------------------------------------
def test_tilt_softens_the_country_share():
    assert tilt(sector_share=0.30, country_share=0.60) == pytest.approx(0.40)   # 0.60 / 1.5
    assert tilt(sector_share=0.50, country_share=0.60) == pytest.approx(0.50)


# --- bands --------------------------------------------------------------------------
@pytest.mark.parametrize("score, expected", [
    (0, "Robust"), (33, "Robust"), (33.4, "Robust"), (34, "Moderate"),
    (66, "Moderate"), (67, "Fragile"), (100, "Fragile"),
])
def test_bands(score, expected):
    assert band(score) == expected


# --- the whole score ----------------------------------------------------------------
def test_all_calm_scores_zero_all_alarm_scores_100():
    assert fragility_score(ALL_CALM).score == 0 and fragility_score(ALL_CALM).band == "Robust"
    assert fragility_score(ALL_ALARM).score == 100


def test_worked_example():
    # Top-10 45%:    100 * (0.45 - 0.15) / 0.45 = 66.67  x 0.30 = 20.00
    # Tilt 60%:      100 * (0.60 - 0.25) / 0.45 = 77.78  x 0.20 = 15.56
    # Stress corr .85: 100 * (0.85 - 0.30) / 0.60 = 91.67 x 0.25 = 22.92
    # Crash 24%:     100 * (0.24 - 0.10) / 0.30 = 46.67  x 0.25 = 11.67
    #                                                     total = 70.14 -> Fragile
    result = fragility_score({"company_concentration": 0.45, "sector_country_tilt": 0.60,
                              "moves_together": 0.85, "crash_loss": 0.24})
    assert result.score == pytest.approx(70.14, abs=0.01)
    assert result.band == "Fragile"
    assert result.top_reason == "Holdings fall together on bad weeks"


def test_breakdown_adds_up_to_the_score():
    result = fragility_score({"company_concentration": 0.30, "sector_country_tilt": 0.40,
                              "moves_together": 0.50, "crash_loss": 0.20})
    assert result.breakdown["points"].sum() == pytest.approx(result.score)


def test_missing_check_is_left_out_and_weights_rescaled():
    # One fund: "moves together" can't be measured. The other three weights
    # (0.30, 0.20, 0.25 = 0.75) are scaled up to 0.40, 0.267, 0.333.
    metrics = dict(ALL_ALARM, moves_together=float("nan"))
    result = fragility_score(metrics)
    assert result.score == pytest.approx(100)
    row = result.breakdown.set_index("component").loc["moves_together"]
    assert not row["available"] and math.isnan(row["points"])
    assert result.breakdown["used_weight"].sum() == pytest.approx(1.0)


def test_nothing_measurable_gives_unknown():
    result = fragility_score({})
    assert math.isnan(result.score) and result.band == "Unknown"


def test_score_breakdown_biggest_first():
    result = fragility_score({"company_concentration": 0.60, "sector_country_tilt": 0.25,
                              "moves_together": float("nan"), "crash_loss": 0.25})
    table = score_breakdown(result)
    assert table.loc[0, "label"] == "Too much money in a few companies"
    assert not table.iloc[-1]["available"]


# --- glue from the other modules ---------------------------------------------------
def test_metrics_from_module_results():
    conc = SimpleNamespace(top10_share=0.45, top_sector_share=0.30, top_country_share=0.90)
    corr = SimpleNamespace(avg_stress_corr=0.85)
    m = metrics_from(conc, corr, worst_loss=-0.24)
    assert m == {"company_concentration": 0.45, "sector_country_tilt": pytest.approx(0.60),
                 "moves_together": 0.85, "crash_loss": 0.24}