from __future__ import annotations

import httpx

from glycoguard.gemini_audit import (
    DEFAULT_GEMINI_MODEL,
    audit_prediction_correctness,
    load_gemini_audit_config,
    payload_from_report,
    review_with_gemini,
)
from glycoguard.risk import classify_risk
from glycoguard.schemas import CGMInput


def test_load_gemini_audit_config_dedupes_keys_and_defaults(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY_1", "key-a")
    monkeypatch.setenv("GEMINI_API_KEY_2", "key-b")
    monkeypatch.setenv("GEMINI_API_KEY_3", "key-a")
    monkeypatch.setenv("GEMINI_API_KEY", "key-c")

    config = load_gemini_audit_config()

    assert config.model == DEFAULT_GEMINI_MODEL
    assert config.api_keys == ["key-a", "key-b", "key-c"]


def test_payload_from_report_uses_recent_trace_and_override() -> None:
    report = {
        "patient_id": "demo-001",
        "recent_trace": [
            {"timestamp": f"2026-04-06T00:{index:02d}:00", "glucose": float(90 + index)}
            for index in range(30)
        ],
        "context": {
            "carbs_1h": 10.0,
            "carbs_2h": 20.0,
            "insulin_on_board": 3.0,
            "activity": 0.4,
            "sleep_flag": 1,
            "stress_score": 0.3,
        },
    }

    payload = payload_from_report(report, current_glucose=101.0)

    assert payload.patient_id == "demo-001"
    assert len(payload.glucose_readings) == 24
    assert payload.glucose_readings[-1] == 101.0
    assert payload.carbs_last_hour == 10.0
    assert payload.insulin_on_board == 3.0


def test_audit_prediction_correctness_matches_deterministic_risk() -> None:
    payload = CGMInput(
        patient_id="demo-001",
        glucose_readings=[125.0] * 20 + [118.0, 112.0, 106.0, 100.0],
        carbs_last_hour=0.0,
        carbs_last_2h=0.0,
        insulin_on_board=2.0,
        activity_level=0.2,
        sleep_flag=0,
        stress_score=0.1,
    )
    roc_15 = float(payload.glucose_readings[-1] - payload.glucose_readings[-4])
    risk = classify_risk(
        prob=0.52,
        predicted_glucose_30min=76.0,
        current_glucose=float(payload.glucose_readings[-1]),
        roc_15=roc_15,
    )
    prediction = {
        "status": "ok",
        "hypo_probability": 0.52,
        "risk_level": risk["risk_level"],
        "alert_required": risk["alert_required"],
        "watch_buzz": risk["watch_buzz"],
        "prob_risk": risk["prob_risk"],
        "forecast_risk": risk["forecast_risk"],
        "predicted_glucose_30min": 76.0,
    }

    audit = audit_prediction_correctness(payload, prediction)

    assert audit["is_consistent"] is True
    assert audit["mismatches"] == []
    assert audit["expected"]["risk_level"] == risk["risk_level"]


def test_review_with_gemini_rotates_keys_until_success(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("GEMINI_API_KEY_1", "key-a")
    monkeypatch.setenv("GEMINI_API_KEY_2", "key-b")
    monkeypatch.setenv("GEMINI_API_KEY_3", "key-c")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    calls: list[str] = []

    class FakeResponse:
        def __init__(self, status_code: int, payload: dict[str, object] | None = None, text: str = "") -> None:
            self.status_code = status_code
            self._payload = payload or {}
            self.text = text

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                request = httpx.Request("POST", "https://example.com")
                response = httpx.Response(self.status_code, request=request)
                raise httpx.HTTPStatusError("error", request=request, response=response)

        def json(self) -> dict[str, object]:
            return self._payload

    class FakeClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
            return None

        def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]) -> FakeResponse:  # type: ignore[override]
            del url, json
            calls.append(headers["x-goog-api-key"])
            if len(calls) < 3:
                return FakeResponse(429, text="rate limited")
            return FakeResponse(
                200,
                payload={
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": '{"verdict":"PASS","summary":"Consistent","mismatches":[],"ui_notes":[]}'
                                    }
                                ]
                            }
                        }
                    ]
                },
            )

    monkeypatch.setattr("glycoguard.gemini_audit.httpx.Client", FakeClient)

    payload = CGMInput(
        patient_id="demo-001",
        glucose_readings=[120.0] * 24,
        carbs_last_hour=0.0,
        carbs_last_2h=0.0,
        insulin_on_board=0.0,
        activity_level=0.0,
        sleep_flag=0,
        stress_score=0.0,
    )
    prediction = {
        "status": "ok",
        "hypo_probability": 0.1,
        "risk_level": "LOW",
        "alert_required": False,
        "watch_buzz": False,
        "prob_risk": "LOW",
        "forecast_risk": "LOW",
        "predicted_glucose_30min": 110.0,
    }
    audit = audit_prediction_correctness(payload, prediction)

    result = review_with_gemini(payload, prediction, audit)

    assert calls == ["key-a", "key-b", "key-c"]
    assert result["key_slot"] == 3
    assert "PASS" in result["response_text"]
