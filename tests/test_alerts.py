"""Tests for fragility.alerts: every rule fires once and stays quiet once."""

import re
from types import SimpleNamespace

import pandas as pd
import pytest

from fragility.alerts import (
    HIGH,
    MEDIUM,
    POSITIVE,
    a1_hidden_company,
    a2_top_heavy,
    a3_duplicate_funds,
    a4_sector_tilt,
    a5_country_tilt,
    a6_moves_as_one,
    a7_crash_loss,
    a8_illusion_of_choice,
    build_alerts,
    high_level,
    split_alerts,
)
from fragility.config import ALERT_THRESHOLDS as T
from fragility.config import DEMO_DIR, PRICE_CACHE


# --- a fake Report: calm by default, each test turns one thing up ---------------------
def report(**changes):
    companies = pd.DataFrame([
        # holding_id, company, exposure, exposure_gbp, via, n_routes, kind
        ["APPLE", "Apple Inc", 0.05, 500, "VWRL.L", 1, "company"],
        ["SHELL", "Shell PLC", 0.03, 300, "VWRL.L", 1, "company"],
        ["RESIDUAL", "Other (residual)", 0.92, 9200, "VWRL.L", 1, "residual"],
    ], columns=["holding_id", "company", "exposure", "exposure_gbp", "via", "n_routes",
                "kind"])
    base = dict(
        total_value=10_000.0,
        portfolio=pd.DataFrame({"ticker": ["VWRL.L", "VUSA.L"],
                                "value_gbp": [6000.0, 4000.0]}),
        companies=companies,
        overlap=pd.DataFrame([[1.0, 0.3], [0.3, 1.0]], index=["VWRL.L", "VUSA.L"],
                             columns=["VWRL.L", "VUSA.L"]),
        concentration=SimpleNamespace(top10_share=0.25, top_sector="Information Technology",
                                      top_sector_share=0.25, top_country="United States",
                                      top_country_share=0.50),
        correlation=SimpleNamespace(avg_stress_corr=0.50, effective_groups=3.0),
        worst=SimpleNamespace(name="Covid crash", worst_loss=-0.20,
                              start=pd.Timestamp("2020-02-19"),
                              end=pd.Timestamp("2020-03-23")),
        score=SimpleNamespace(score=45.0),
    )
    base.update(changes)
    return SimpleNamespace(**base)


def big_apple():
    c = report().companies.copy()
    c.loc[0, ["exposure", "exposure_gbp", "via", "n_routes"]] = [
        0.13, 1300, "EQQQ.L, VWRL.L, direct", 3]
    return c


def conc(**kw):
    base = vars(report().concentration).copy()
    base.update(kw)
    return SimpleNamespace(**base)


CALM = report()


# --- each rule: fires, and stays quiet ------------------------------------------------
def test_a1_fires_for_a_big_company_through_several_routes():
    a = a1_hidden_company(report(companies=big_apple()), T)
    assert a.id == "A1" and a.severity == HIGH          # 13% >= 1.5 x 8%
    assert a.message == ("Apple Inc is 13% of your money (£1,300), through 3 different "
                         "routes: EQQQ.L, VWRL.L, direct.")


def test_a1_quiet_when_big_but_one_route():
    c = big_apple()
    c.loc[0, "n_routes"] = 1
    assert a1_hidden_company(report(companies=c), T) is None
    assert a1_hidden_company(CALM, T) is None


def test_a2_top_heavy():
    a = a2_top_heavy(report(concentration=conc(top10_share=0.45)), T)
    assert a.severity == MEDIUM and "45% of your money sits in just 10 companies" in a.message
    assert a2_top_heavy(CALM, T) is None


def test_a3_duplicate_funds_names_the_pair():
    m = pd.DataFrame([[1.0, 0.92], [0.92, 1.0]], index=["VUSA.L", "CSPX.L"],
                     columns=["VUSA.L", "CSPX.L"])
    a = a3_duplicate_funds(report(overlap=m), T)
    assert a.evidence["pair"] == ("VUSA.L", "CSPX.L") and a.severity == HIGH
    assert "VUSA.L and CSPX.L hold largely the same companies (92% overlap)" in a.message
    assert a3_duplicate_funds(CALM, T) is None


def test_a4_sector_tilt_with_pound_cost():
    a = a4_sector_tilt(report(concentration=conc(top_sector_share=0.41)), T)
    # 41% x 20% x £10,000 = £820
    assert a.message == ("41% of your money is in Information Technology. A 20% fall in "
                         "that sector would cost about £820.")
    assert a4_sector_tilt(CALM, T) is None


def test_a5_country_tilt():
    a = a5_country_tilt(report(concentration=conc(top_country_share=0.78)), T)
    assert "78% of your money depends on companies in United States" in a.message
    assert a5_country_tilt(CALM, T) is None


def test_a6_moves_as_one():
    a = a6_moves_as_one(report(correlation=SimpleNamespace(avg_stress_corr=0.92,
                                                          effective_groups=1.2)), T)
    assert a.severity == HIGH                          # 0.92 >= halfway from 0.80 to 1
    assert a6_moves_as_one(CALM, T) is None


def test_a6_quiet_when_not_measurable():
    one_fund = report(correlation=SimpleNamespace(avg_stress_corr=float("nan"),
                                                  effective_groups=1.0))
    assert a6_moves_as_one(one_fund, T) is None


def test_a7_crash_loss_in_pounds_and_weeks():
    worst = SimpleNamespace(name="Covid crash", worst_loss=-0.32,
                            start=pd.Timestamp("2020-02-19"), end=pd.Timestamp("2020-03-23"))
    a = a7_crash_loss(report(worst=worst), T)
    assert a.message == ("In the Covid crash (Feb 2020), this portfolio would have lost "
                         "32% (£3,200) in about 5 weeks.")
    assert a7_crash_loss(CALM, T) is None


def test_a8_illusion_of_choice():
    seven = pd.DataFrame({"ticker": list("ABCDEFG"), "value_gbp": [1.0] * 7})
    a = a8_illusion_of_choice(
        report(portfolio=seven,
               correlation=SimpleNamespace(avg_stress_corr=0.9, effective_groups=1.4)), T)
    assert a.message == "You hold 7 investments, but they move like 1.4 groups."
    assert a8_illusion_of_choice(CALM, T) is None   # only 2 holdings


# --- severity --------------------------------------------------------------------------
def test_high_level_is_one_and_a_half_times_or_halfway_to_100():
    assert high_level(0.08, cap=None) == pytest.approx(0.12)
    assert high_level(0.40) == pytest.approx(0.60)
    assert high_level(0.80) == pytest.approx(0.90)      # 1.2 is impossible -> halfway


# --- putting it together ---------------------------------------------------------------
def test_nothing_fires_gives_the_positive_card():
    alerts = build_alerts(CALM, T)
    assert [a.id for a in alerts] == ["P1"] and alerts[0].severity == POSITIVE


def test_low_score_adds_positive_card_even_with_a_minor_alert():
    r = report(concentration=conc(top10_share=0.42), score=SimpleNamespace(score=30.0))
    assert [a.id for a in build_alerts(r, T)] == ["A2", "P1"]


def test_most_serious_first_then_biggest_impact():
    r = report(companies=big_apple(),                                  # A1 high
               concentration=conc(top10_share=0.45, top_sector_share=0.40),  # A2, A4 medium
               worst=SimpleNamespace(name="Covid crash", worst_loss=-0.50,
                                     start=pd.Timestamp("2020-02-19"),
                                     end=pd.Timestamp("2020-03-23")))  # A7 high
    ids = [a.id for a in build_alerts(r, T)]
    assert ids[:2] == ["A7", "A1"]        # both high: £5,000 crash beats £1,300 Apple
    assert set(ids[2:]) == {"A2", "A4"}


def test_only_three_shown():
    r = report(companies=big_apple(),
               concentration=conc(top10_share=0.45, top_sector_share=0.40,
                                  top_country_share=0.90))
    shown, more = split_alerts(build_alerts(r, T))
    assert len(shown) == 3 and len(more) >= 1


def test_no_alert_gives_advice():
    r = report(companies=big_apple(),
               concentration=conc(top10_share=0.5, top_sector_share=0.5,
                                  top_country_share=0.9),
               correlation=SimpleNamespace(avg_stress_corr=0.95, effective_groups=1.1))
    advice = re.compile(r"\b(sell|buy|should|must|recommend)\b", re.IGNORECASE)
    assert not any(advice.search(a.message) for a in build_alerts(r, T))


# --- real data: the overlapping demo -----------------------------------------------------
@pytest.mark.skipif(not PRICE_CACHE.exists(), reason="no price cache yet")
def test_overlapping_demo_raises_the_hidden_company_and_sector_alerts():
    from fragility.pipeline import analyse_csv
    ids = {a.id for a in analyse_csv(DEMO_DIR / "overlapping.csv",
                                     max_price_age_days=365).alerts}
    assert {"A1", "A4"} <= ids, ids
    # A3 (VUSA vs CSPX) needs fuller holdings than the top 50: see the change log.