from __future__ import annotations

import pandas as pd

from glycoguard.service import GlycoGuardService


def test_build_watch_payload_falls_back_to_computed_trend_when_prediction_trend_is_missing() -> None:
    service = object.__new__(GlycoGuardService)

    payload = service.build_watch_payload(
        patient_id="demo-patient",
        prediction={
            "status": "insufficient_confidence",
            "roc_15": None,
            "risk_level": None,
            "alert_required": False,
            "watch_buzz": False,
            "predicted_glucose_30min": None,
            "abstention_reason": "Prediction unavailable",
            "watch_status": "Prediction unavailable",
        },
        current_glucose=92.0,
        roc_15=-2.5,
        timestamp=pd.Timestamp("2026-04-06T06:15:00"),
    )

    assert payload["roc_15"] == -2.5
    assert payload["trend"] == "-2.5 mg/dL per 15min"
    assert payload["risk"] == "UNKNOWN"
