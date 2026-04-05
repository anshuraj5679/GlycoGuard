from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from glycoguard.risk import classify_risk
from glycoguard.schemas import CGMInput


DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"


@dataclass(slots=True)
class GeminiAuditConfig:
    model: str
    api_keys: list[str]


def load_gemini_audit_config() -> GeminiAuditConfig:
    keys = [
        value
        for value in (
            os.getenv("GEMINI_API_KEY_1", "").strip(),
            os.getenv("GEMINI_API_KEY_2", "").strip(),
            os.getenv("GEMINI_API_KEY_3", "").strip(),
            os.getenv("GEMINI_API_KEY", "").strip(),
        )
        if value
    ]
    deduped_keys: list[str] = []
    for key in keys:
        if key not in deduped_keys:
            deduped_keys.append(key)
    model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
    return GeminiAuditConfig(model=model, api_keys=deduped_keys)


def payload_from_report(report: dict[str, object], current_glucose: float | None = None) -> CGMInput:
    readings = [float(item["glucose"]) for item in report["recent_trace"][-24:]]
    if current_glucose is not None:
        readings[-1] = float(current_glucose)
    context = dict(report.get("context") or {})
    return CGMInput(
        patient_id=str(report["patient_id"]),
        glucose_readings=readings,
        carbs_last_hour=float(context.get("carbs_1h", 0.0)),
        carbs_last_2h=float(context.get("carbs_2h", 0.0)),
        insulin_on_board=float(context.get("insulin_on_board", 0.0)),
        activity_level=float(context.get("activity", 0.0)),
        sleep_flag=int(context.get("sleep_flag", 0)),
        stress_score=float(context.get("stress_score", 0.0)),
    )


def payload_from_file(path: str | Path, current_glucose: float | None = None) -> CGMInput:
    raw_payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if current_glucose is not None:
        readings = list(raw_payload.get("glucose_readings") or [])
        if readings:
            readings[-1] = float(current_glucose)
            raw_payload["glucose_readings"] = readings
    return CGMInput(**raw_payload)


def audit_prediction_correctness(payload: CGMInput, prediction: dict[str, object]) -> dict[str, object]:
    current_glucose = float(payload.glucose_readings[-1])
    roc_15 = float(payload.glucose_readings[-1] - payload.glucose_readings[-4]) if len(payload.glucose_readings) >= 4 else 0.0

    if prediction.get("status") != "ok":
        return {
            "is_consistent": prediction.get("status") == "insufficient_confidence",
            "status": prediction.get("status"),
            "expected": None,
            "actual": {
                "risk_level": prediction.get("risk_level"),
                "alert_required": prediction.get("alert_required"),
                "watch_buzz": prediction.get("watch_buzz"),
                "predicted_glucose_30min": prediction.get("predicted_glucose_30min"),
                "abstention_reason": prediction.get("abstention_reason"),
            },
            "mismatches": [],
            "notes": ["Prediction abstained; deterministic risk audit skipped."],
        }

    expected = classify_risk(
        prob=float(prediction.get("hypo_probability") or 0.0),
        predicted_glucose_30min=None
        if prediction.get("predicted_glucose_30min") is None
        else float(prediction["predicted_glucose_30min"]),
        current_glucose=current_glucose,
        roc_15=roc_15,
    )

    actual = {
        "current_glucose": round(current_glucose, 1),
        "roc_15": round(roc_15, 1),
        "risk_level": prediction.get("risk_level"),
        "alert_required": bool(prediction.get("alert_required")),
        "watch_buzz": bool(prediction.get("watch_buzz", prediction.get("alert_required"))),
        "prob_risk": prediction.get("prob_risk"),
        "forecast_risk": prediction.get("forecast_risk"),
        "predicted_glucose_30min": prediction.get("predicted_glucose_30min"),
    }

    mismatches: list[dict[str, object]] = []
    for field in ("risk_level", "alert_required", "watch_buzz", "prob_risk", "forecast_risk", "predicted_glucose_30min"):
        if actual.get(field) != expected.get(field):
            mismatches.append(
                {
                    "field": field,
                    "expected": expected.get(field),
                    "actual": actual.get(field),
                }
            )

    return {
        "is_consistent": not mismatches,
        "status": prediction.get("status"),
        "expected": {
            **expected,
            "current_glucose": round(current_glucose, 1),
            "roc_15": round(roc_15, 1),
        },
        "actual": actual,
        "mismatches": mismatches,
        "notes": [],
    }


def build_gemini_review_prompt(payload: CGMInput, prediction: dict[str, object], audit: dict[str, object]) -> str:
    prompt = {
        "task": "Review whether this hypoglycaemia prediction output is internally consistent with the provided deterministic audit.",
        "instructions": [
            "Do not invent medical facts beyond the supplied data.",
            "Treat the deterministic audit as the source of truth for rule correctness.",
            "Respond as compact JSON with keys: verdict, summary, mismatches, ui_notes.",
            "verdict must be one of PASS, FAIL, REVIEW.",
        ],
        "input_payload": payload.model_dump(),
        "model_prediction": prediction,
        "deterministic_audit": audit,
    }
    return json.dumps(prompt, indent=2)


def review_with_gemini(
    payload: CGMInput,
    prediction: dict[str, object],
    audit: dict[str, object],
    *,
    model: str | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, object]:
    config = load_gemini_audit_config()
    selected_model = model or config.model
    if not config.api_keys:
        raise RuntimeError("No Gemini API keys found. Set GEMINI_API_KEY_1..3 or GEMINI_API_KEY in .env.")

    prompt = build_gemini_review_prompt(payload, prediction, audit)
    last_error: str | None = None
    for index, api_key in enumerate(config.api_keys, start=1):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{selected_model}:generateContent"
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 800,
            },
        }
        try:
            with httpx.Client(timeout=timeout_seconds) as client:
                response = client.post(
                    url,
                    headers={
                        "x-goog-api-key": api_key,
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
            if response.status_code in {401, 403, 429, 500, 503}:
                last_error = f"key_{index}: {response.status_code} {response.text[:200]}"
                continue
            response.raise_for_status()
            payload_json = response.json()
            candidates = payload_json.get("candidates", [])
            text = ""
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(str(part.get("text", "")) for part in parts).strip()
            return {
                "model": selected_model,
                "key_slot": index,
                "response_text": text,
                "raw": payload_json,
            }
        except httpx.HTTPError as exc:
            last_error = f"key_{index}: {type(exc).__name__}: {exc}"
            continue

    raise RuntimeError(f"All Gemini keys failed for model {selected_model}. Last error: {last_error}")


def run_prediction_audit(
    service: Any,
    *,
    patient_id: str | None = None,
    payload: CGMInput | None = None,
    payload_file: str | None = None,
    current_glucose: float | None = None,
    use_gemini: bool = False,
    model: str | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, object]:
    if payload is not None:
        audit_payload = payload.model_copy(deep=True)
        if current_glucose is not None:
            readings = list(audit_payload.glucose_readings)
            readings[-1] = float(current_glucose)
            audit_payload = audit_payload.model_copy(update={"glucose_readings": readings})
    elif payload_file:
        audit_payload = payload_from_file(payload_file, current_glucose=current_glucose)
    else:
        report = service.get_report(patient_id=patient_id)
        audit_payload = payload_from_report(report, current_glucose=current_glucose)

    prediction = service.explain(audit_payload)
    prediction.pop("feature_frame", None)
    audit = audit_prediction_correctness(audit_payload, prediction)
    result: dict[str, object] = {
        "payload": audit_payload.model_dump(mode="json"),
        "prediction": prediction,
        "audit": audit,
        "gemini_review": None,
    }
    if use_gemini:
        result["gemini_review"] = review_with_gemini(
            audit_payload,
            prediction,
            audit,
            model=model,
            timeout_seconds=timeout_seconds,
        )
    return result
