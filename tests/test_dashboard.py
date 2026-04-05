from __future__ import annotations

import pytest

from glycoguard.dashboard.app import _daily_guidance, _friendly_top_factors, parse_glucose_readings


def test_parse_glucose_readings_accepts_comma_separated_values() -> None:
    values = parse_glucose_readings(",".join(str(90 + idx) for idx in range(24)))
    assert len(values) == 24
    assert values[0] == 90.0
    assert values[-1] == 113.0


def test_parse_glucose_readings_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        parse_glucose_readings("90, 91, 92")


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
