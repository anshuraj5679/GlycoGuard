from __future__ import annotations

RISK_PRIORITY = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def classify_probability_risk(
    prob: float,
    *,
    high_threshold: float = 0.7,
    medium_threshold: float = 0.4,
) -> str:
    if prob >= high_threshold:
        return "HIGH"
    if prob >= medium_threshold:
        return "MEDIUM"
    return "LOW"


def classify_forecast_risk(predicted_glucose_30min: float | None) -> str:
    if predicted_glucose_30min is None:
        return "LOW"
    if predicted_glucose_30min < 70.0:
        return "HIGH"
    if predicted_glucose_30min < 80.0:
        return "MEDIUM"
    return "LOW"


def classify_trend_risk(roc_15: float, predicted_glucose_30min: float | None) -> str:
    if predicted_glucose_30min is None:
        return "LOW"
    if roc_15 <= -2.0 and predicted_glucose_30min < 80.0:
        return "HIGH"
    if roc_15 <= -1.0 and predicted_glucose_30min < 90.0:
        return "MEDIUM"
    return "LOW"


def _max_risk(*levels: str) -> str:
    return max(levels, key=lambda level: RISK_PRIORITY[level])


def classify_risk(
    *,
    prob: float,
    predicted_glucose_30min: float | None,
    current_glucose: float,
    roc_15: float,
    high_threshold: float = 0.7,
    medium_threshold: float = 0.4,
    severe_threshold: float = 54.0,
) -> dict[str, object]:
    prob_risk = classify_probability_risk(prob, high_threshold=high_threshold, medium_threshold=medium_threshold)
    forecast_risk = classify_forecast_risk(predicted_glucose_30min)
    trend_risk = classify_trend_risk(roc_15, predicted_glucose_30min)

    final_risk = _max_risk(prob_risk, forecast_risk, trend_risk)
    if current_glucose <= severe_threshold:
        final_risk = "HIGH"

    alert_required = final_risk == "HIGH"
    return {
        "risk_level": final_risk,
        "alert_required": alert_required,
        "watch_buzz": alert_required,
        "prob_risk": prob_risk,
        "forecast_risk": _max_risk(forecast_risk, trend_risk),
        "trend_risk": trend_risk,
        "hypo_probability": round(float(prob), 4),
        "predicted_glucose_30min": None if predicted_glucose_30min is None else round(float(predicted_glucose_30min), 1),
    }
