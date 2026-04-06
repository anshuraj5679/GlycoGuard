from __future__ import annotations

import pytest

from glycoguard.dashboard.app import (
    _active_watch_payload,
    _daily_guidance,
    _display_risk_score,
    _forecast_only_figure,
    _friendly_top_factors,
    _profile_seed,
    _watch_preview_html,
    _watch_preview_payload,
    compose_glucose_readings,
    parse_glucose_history,
    parse_glucose_readings,
)


def test_parse_glucose_readings_accepts_comma_separated_values() -> None:
    values = parse_glucose_readings(",".join(str(90 + idx) for idx in range(24)))
    assert len(values) == 24
    assert values[0] == 90.0
    assert values[-1] == 113.0


def test_parse_glucose_readings_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        parse_glucose_readings("90, 91, 92")


def test_parse_glucose_history_accepts_23_values() -> None:
    values = parse_glucose_history(",".join(str(90 + idx) for idx in range(23)))
    assert len(values) == 23
    assert values[0] == 90.0
    assert values[-1] == 112.0


def test_parse_glucose_history_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        parse_glucose_history(",".join(str(90 + idx) for idx in range(24)))


def test_compose_glucose_readings_appends_current_for_23_value_history() -> None:
    values = compose_glucose_readings([float(90 + idx) for idx in range(23)], 140.0)
    assert len(values) == 24
    assert values[-2] == 112.0
    assert values[-1] == 140.0


def test_compose_glucose_readings_replaces_last_value_for_loaded_trace() -> None:
    values = compose_glucose_readings([float(90 + idx) for idx in range(24)], 141.0)
    assert len(values) == 24
    assert values[-2] == 112.0
    assert values[-1] == 141.0


def test_daily_guidance_returns_plain_language_copy() -> None:
    prediction = {"risk_level": "HIGH", "predicted_glucose_30min": 63.2}
    guidance = _daily_guidance(prediction)
    assert "High risk" in guidance["headline"]
    assert "63" in guidance["body"]


def test_friendly_top_factors_uses_human_labels() -> None:
    prediction = {
        "top_factors": [
            {"feature": "insulin_on_board", "contribution": 0.8},
            {"feature": "carbs_1h", "contribution": -0.2},
        ]
    }
    factors = _friendly_top_factors(prediction)
    assert "Active insulin" in factors[0]
    assert "Carbs in the last hour" in factors[1]


def test_profile_seed_returns_patient_defaults() -> None:
    seed = _profile_seed(None)
    assert seed["diabetes_type"] == "Type 1"
    assert seed["target_range_low"] == 70.0
    assert seed["target_range_high"] == 180.0


def test_watch_preview_payload_returns_placeholder_before_live_feed() -> None:
    payload = _watch_preview_payload(None)
    assert payload["risk"] == "UNKNOWN"
    assert payload["forecast_30min"] is None
    assert "validated risk alerts" in payload["reason"]


def test_daily_guidance_handles_suppressed_forecast_for_active_low() -> None:
    prediction = {
        "status": "ok",
        "risk_level": "HIGH",
        "predicted_glucose_30min": None,
        "forecast_notice": "Current glucose is already below 54 mg/dL.",
    }
    guidance = _daily_guidance(prediction)
    assert "Treat the current low glucose now" in guidance["headline"]
    assert "below 54" in guidance["body"]


def test_active_watch_payload_uses_live_prediction_state() -> None:
    payload = _active_watch_payload(
        readings=[110.0 + idx for idx in range(20)] + [128.0, 126.0, 124.0, 122.0],
        prediction={
            "status": "ok",
            "patient_id": "manual-user",
            "risk_level": "HIGH",
            "alert_required": True,
            "watch_buzz": True,
            "predicted_glucose_30min": 68.0,
            "top_reason": "Glucose dropping 6.0 mg/dL per 15 min",
            "forecast_warning": "Predicted to cross 70 mg/dL threshold",
            "watch_status": "Watch buzz triggered - eat 15g carbs now",
        },
    )
    assert payload["patient_id"] == "manual-user"
    assert payload["risk"] == "HIGH"
    assert payload["buzz"] is True
    assert "Glucose dropping" in payload["reason"]
    assert payload["forecast_warning"] == "Predicted to cross 70 mg/dL threshold"
    assert payload["watch_status"] == "Watch buzz triggered - eat 15g carbs now"
    assert payload["display_risk_score"] == 0.85


def test_display_risk_score_reflects_forecast_and_trend_when_probability_is_low() -> None:
    score = _display_risk_score(
        {
            "status": "ok",
            "hypo_probability": 0.0,
            "risk_level": "MEDIUM",
            "predicted_glucose_30min": 76.0,
            "roc_15": -1.4,
            "current_glucose": 95.0,
        }
    )

    assert score == 0.6


def test_forecast_only_figure_hides_recent_glucose_line() -> None:
    figure = _forecast_only_figure(
        current_glucose=122.0,
        prediction={
            "forecast_trace": [118.0, 114.0, 110.0, 106.0, 102.0, 98.0],
            "forecast_lower": [110.0, 106.0, 102.0, 98.0, 94.0, 90.0],
            "forecast_upper": [126.0, 122.0, 118.0, 114.0, 110.0, 106.0],
        },
    )
    trace_names = [trace.name for trace in figure.data]
    assert "Recent glucose" not in trace_names
    assert "Current glucose" in trace_names
    assert "30-minute forecast" in trace_names
    current_trace = next(trace for trace in figure.data if trace.name == "Current glucose")
    forecast_trace = next(trace for trace in figure.data if trace.name == "30-minute forecast")
    assert current_trace.marker.color == "#15222e"
    assert forecast_trace.line.color == "#10B981"


def test_watch_preview_html_is_dedented_and_escapes_user_text() -> None:
    html = _watch_preview_html(
        {
            "glucose": 122.0,
            "roc_15": -4.0,
            "trend": "-4.0 mg/dL per 15min",
            "risk": "HIGH",
            "reason": "Glucose <dropping> fast",
            "buzz": True,
            "forecast_30min": 68.0,
            "forecast_warning": "Predicted to cross 70 mg/dL threshold",
            "hypo_probability": 0.82,
            "top_reason": "Glucose <dropping> fast",
            "watch_status": "Watch buzz triggered <now>",
        }
    )

    assert html.startswith("<div")
    assert '<div style="display:flex;gap:0.35rem;">' in html
    assert "&lt;dropping&gt;" in html
    assert "&lt;now&gt;" in html
