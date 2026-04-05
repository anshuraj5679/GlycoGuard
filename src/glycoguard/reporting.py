from __future__ import annotations

import numpy as np
import pandas as pd


def build_agp_payload(df: pd.DataFrame, days: int = 14) -> dict[str, object]:
    if df.empty:
        return {"summary": {}, "percentiles": []}

    end_time = df.index.max()
    start_time = end_time - pd.Timedelta(days=days)
    window = df.loc[df.index >= start_time]
    clock = window.index.strftime("%H:%M")
    grouped = window.groupby(clock)["glucose"]

    profile = pd.DataFrame(
        {
            "time": sorted(grouped.groups.keys()),
        }
    )
    quantiles = {
        "p05": grouped.quantile(0.05),
        "p25": grouped.quantile(0.25),
        "p50": grouped.quantile(0.50),
        "p75": grouped.quantile(0.75),
        "p95": grouped.quantile(0.95),
    }
    for name, series in quantiles.items():
        profile[name] = [float(series.loc[item]) for item in profile["time"]]

    summary = {
        "mean_glucose": float(window["glucose"].mean()),
        "time_in_range": float(window["glucose"].between(70, 180).mean()),
        "time_below_range": float((window["glucose"] < 70).mean()),
        "time_above_range": float((window["glucose"] > 180).mean()),
        "cv": float(window["glucose"].std(ddof=0) / max(window["glucose"].mean(), 1.0)),
        "lowest_glucose": float(window["glucose"].min()),
        "highest_glucose": float(window["glucose"].max()),
    }
    return {"summary": summary, "percentiles": profile.to_dict(orient="records")}


def build_alert_log(feature_df: pd.DataFrame, probabilities: np.ndarray, limit: int = 20) -> list[dict[str, object]]:
    frame = feature_df.copy()
    frame["hypo_probability"] = probabilities
    frame = frame.loc[(frame["hypo_probability"] >= 0.4) | (frame["hypo_label"] == 1)]
    frame = frame.tail(limit)
    rows: list[dict[str, object]] = []
    for timestamp, row in frame.iterrows():
        risk_level = "HIGH" if row["hypo_probability"] >= 0.7 else "MEDIUM" if row["hypo_probability"] >= 0.4 else "LOW"
        rows.append(
            {
                "timestamp": timestamp.isoformat(),
                "hypo_probability": float(row["hypo_probability"]),
                "risk_level": risk_level,
                "actual_hypo": bool(row["hypo_label"]),
            }
        )
    return rows


def build_waterfall_payload(explanation: dict[str, object]) -> dict[str, object]:
    payload = explanation.get("waterfall", {}) or {}
    return {
        "base_value": float(payload.get("base_value", 0.0)),
        "feature_names": list(payload.get("feature_names", [])),
        "feature_values": dict(payload.get("feature_values", {})),
        "shap_values": dict(payload.get("shap_values", {})),
        "backend": payload.get("backend", "shap"),
    }
