from __future__ import annotations

import json

from glycoguard.live_watch import live_watch_path, read_live_watch_payload, write_live_watch_payload


def test_write_and_read_live_watch_payload_round_trip(tmp_path) -> None:
    payload = {
        "patient_id": "manual-user",
        "glucose": 92.0,
        "roc_15": -6.0,
        "trend": "-6.0 mg/dL per 15min",
        "risk": "HIGH",
        "reason": "Glucose dropping fast",
        "buzz": True,
        "forecast_30min": 68.0,
        "forecast_warning": "Predicted to cross 70 mg/dL threshold",
        "hypo_probability": 0.22,
        "top_reason": "Glucose dropping fast",
        "watch_status": "Watch buzz triggered - eat 15g carbs now",
        "updated_at": "2026-04-06T05:30:00",
        "status": "live",
        "session_id": None,
    }

    path = write_live_watch_payload(tmp_path, payload)

    assert path == live_watch_path(tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["risk"] == "HIGH"
    assert read_live_watch_payload(tmp_path, max_age_seconds=10**9) == payload


def test_read_live_watch_payload_returns_none_when_stale(tmp_path) -> None:
    write_live_watch_payload(
        tmp_path,
        {
            "patient_id": "manual-user",
            "updated_at": "2020-01-01T00:00:00",
        },
    )

    assert read_live_watch_payload(tmp_path, max_age_seconds=60) is None
