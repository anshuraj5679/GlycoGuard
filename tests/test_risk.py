from __future__ import annotations

from glycoguard.risk import classify_risk


def test_demo_scenario_promotes_fast_drop_near_threshold_to_high() -> None:
    risk = classify_risk(
        prob=0.22,
        predicted_glucose_30min=71.0,
        current_glucose=160.0,
        roc_15=-4.0,
    )
    assert risk["risk_level"] == "HIGH"
    assert risk["alert_required"] is True
    assert risk["watch_buzz"] is True


def test_medium_when_forecast_enters_buffer_zone_without_fast_drop() -> None:
    risk = classify_risk(
        prob=0.21,
        predicted_glucose_30min=76.0,
        current_glucose=95.0,
        roc_15=-0.6,
    )
    assert risk["risk_level"] == "MEDIUM"
    assert risk["alert_required"] is False


def test_probability_alone_can_trigger_high() -> None:
    risk = classify_risk(
        prob=0.81,
        predicted_glucose_30min=96.0,
        current_glucose=140.0,
        roc_15=0.2,
    )
    assert risk["risk_level"] == "HIGH"
    assert risk["prob_risk"] == "HIGH"


def test_low_when_probability_and_forecast_are_safe() -> None:
    risk = classify_risk(
        prob=0.18,
        predicted_glucose_30min=118.0,
        current_glucose=120.0,
        roc_15=0.1,
    )
    assert risk["risk_level"] == "LOW"
    assert risk["watch_buzz"] is False
