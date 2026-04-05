from __future__ import annotations

from glycoguard.cli import _build_api_command, _build_dashboard_command, build_parser, run_prediction_audit
from glycoguard.risk import classify_risk


def test_serve_parser_accepts_ports_and_windows_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["serve", "--api-port", "9000", "--dashboard-port", "8601", "--separate-windows"])
    assert args.command == "serve"
    assert args.api_port == 9000
    assert args.dashboard_port == 8601
    assert args.separate_windows is True


def test_serve_commands_target_api_and_dashboard() -> None:
    api_command = _build_api_command("python", 8000)
    dashboard_command = _build_dashboard_command("python", 8501)

    assert api_command[:4] == ["python", "-m", "uvicorn", "glycoguard.api.main:app"]
    assert "--port" in api_command
    assert "8000" in api_command

    assert dashboard_command[:4] == ["python", "-m", "streamlit", "run"]
    assert dashboard_command[-2:] == ["--server.port", "8501"]


def test_audit_parser_accepts_payload_and_gemini_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "audit-prediction",
            "--patient-id",
            "demo-001",
            "--current-glucose",
            "92",
            "--gemini",
            "--model",
            "gemini-3.1-flash-lite",
            "--timeout-seconds",
            "12",
        ]
    )

    assert args.command == "audit-prediction"
    assert args.patient_id == "demo-001"
    assert args.current_glucose == 92
    assert args.gemini is True
    assert args.model == "gemini-3.1-flash-lite"
    assert args.timeout_seconds == 12


def test_run_prediction_audit_uses_report_history_and_current_override() -> None:
    class DummyService:
        def get_report(self, patient_id: str | None = None) -> dict[str, object]:
            return {
                "patient_id": patient_id or "demo-001",
                "recent_trace": [
                    {"timestamp": f"2026-04-06T00:{index:02d}:00", "glucose": float(100 + index)}
                    for index in range(24)
                ],
                "context": {
                    "carbs_1h": 6.0,
                    "carbs_2h": 12.0,
                    "insulin_on_board": 2.5,
                    "activity": 0.2,
                    "sleep_flag": 0,
                    "stress_score": 0.1,
                },
            }

        def explain(self, payload):  # type: ignore[no-untyped-def]
            risk = classify_risk(
                prob=0.25,
                predicted_glucose_30min=84.0,
                current_glucose=float(payload.glucose_readings[-1]),
                roc_15=float(payload.glucose_readings[-1] - payload.glucose_readings[-4]),
            )
            return {
                "patient_id": payload.patient_id,
                "status": "ok",
                "current_glucose": round(float(payload.glucose_readings[-1]), 1),
                "roc_15": round(float(payload.glucose_readings[-1] - payload.glucose_readings[-4]), 1),
                "hypo_probability": 0.25,
                "risk_level": risk["risk_level"],
                "prob_risk": risk["prob_risk"],
                "forecast_risk": risk["forecast_risk"],
                "predicted_glucose_30min": 84.0,
                "alert_required": risk["alert_required"],
                "watch_buzz": risk["watch_buzz"],
                "explanation": "",
                "top_factors": [],
                "shap_values": {},
                "waterfall": None,
                "model_backend": "xgboost",
                "forecast_backend": "tft",
                "confidence": 0.9,
                "abstention_reason": None,
            }

    result = run_prediction_audit(DummyService(), patient_id="demo-001", current_glucose=92.0)

    assert result["payload"]["patient_id"] == "demo-001"
    assert result["payload"]["glucose_readings"][-1] == 92.0
    assert result["prediction"]["current_glucose"] == 92.0
    assert result["audit"]["is_consistent"] is True
    assert result["gemini_review"] is None
