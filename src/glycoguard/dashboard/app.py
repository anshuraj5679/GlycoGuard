from __future__ import annotations

from datetime import datetime
from html import escape
from textwrap import dedent
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None

from glycoguard.schemas import CGMInput
from glycoguard.service import get_service


PALETTE = {
    "bg": "#f5efe6",
    "ink": "#15222e",
    "accent": "#d96c06",
    "high": "#EF4444",
    "medium": "#F97316",
    "low": "#10B981",
    "muted": "#5b6770",
    "panel": "#fffaf4",
}

FRIENDLY_FEATURE_LABELS = {
    "glucose": "Current glucose",
    "roc_15": "15-minute glucose trend",
    "roc_30": "30-minute glucose trend",
    "min_2h": "Lowest recent glucose",
    "mean_30m": "Recent glucose average",
    "mean_2h": "2-hour glucose average",
    "carbs_1h": "Carbs in the last hour",
    "carbs_2h": "Carbs in the last 2 hours",
    "insulin_on_board": "Active insulin",
    "activity": "Recent activity",
    "sleep_flag": "Sleep period",
    "stress_score": "Stress level",
    "lbgi_2h": "Low-glucose risk index",
    "hbgi_2h": "High-glucose risk index",
    "is_night": "Night-time pattern",
    "hour_sin": "Time-of-day pattern",
    "hour_cos": "Time-of-day pattern",
}


def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(217,108,6,0.12), transparent 26%),
                linear-gradient(180deg, {PALETTE["bg"]} 0%, #fcf8f2 100%);
            color: {PALETTE["ink"]};
            font-family: Georgia, "Times New Roman", serif;
        }}
        .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }}
        .hero-card {{
            background: linear-gradient(135deg, #fff8ee 0%, #fffdf8 100%);
            border: 1px solid rgba(21,34,46,0.08);
            border-radius: 20px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 16px 40px rgba(21,34,46,0.08);
        }}
        .metric-chip {{
            border-radius: 14px;
            padding: 0.9rem 1rem;
            background: {PALETTE["panel"]};
            border: 1px solid rgba(21,34,46,0.08);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _risk_color(risk_level: str | None) -> str:
    if risk_level == "HIGH":
        return PALETTE["high"]
    if risk_level == "MEDIUM":
        return PALETTE["medium"]
    if risk_level in {None, "UNKNOWN"}:
        return PALETTE["muted"]
    return PALETTE["low"]


def _glucose_value_color(glucose: float | None) -> str:
    if glucose is None:
        return "#ffffff"
    if glucose < 54.0:
        return PALETTE["high"]
    if glucose < 70.0:
        return PALETTE["medium"]
    if glucose > 180.0:
        return "#F59E0B"
    return "#FFFFFF"


def _chart_glucose_color(glucose: float | None) -> str:
    if glucose is None:
        return PALETTE["ink"]
    if glucose < 54.0:
        return PALETTE["high"]
    if glucose < 70.0:
        return PALETTE["medium"]
    if glucose > 180.0:
        return "#F59E0B"
    return PALETTE["ink"]


def _trend_symbol(roc_15: float) -> str:
    if roc_15 < -2.0:
        return "↓↓"
    if roc_15 < -1.0:
        return "↓"
    if roc_15 > 2.0:
        return "↑↑"
    if roc_15 > 1.0:
        return "↑"
    return "→"


def _trend_color(roc_15: float) -> str:
    if roc_15 < -2.0:
        return PALETTE["high"]
    if roc_15 < -1.0:
        return PALETTE["medium"]
    if roc_15 > 0.0:
        return "#F59E0B"
    return "#FFFFFF"


def _forecast_value_color(forecast: float | None) -> str:
    if forecast is None:
        return "#94a3b8"
    if forecast < 70.0:
        return PALETTE["high"]
    if forecast < 80.0:
        return PALETTE["medium"]
    return "#FFFFFF"


def _chart_forecast_color(forecast: float | None) -> str:
    if forecast is None:
        return PALETTE["muted"]
    if forecast < 70.0:
        return PALETTE["high"]
    if forecast < 80.0:
        return PALETTE["medium"]
    return PALETTE["low"]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        return f"rgba(148,163,184,{alpha})"
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return f"rgba({red},{green},{blue},{alpha})"


def parse_glucose_readings(raw_text: str) -> list[float]:
    tokens = [token.strip() for token in raw_text.replace("\n", ",").split(",")]
    values = [float(token) for token in tokens if token]
    if len(values) != 24:
        raise ValueError("Enter exactly 24 glucose readings separated by commas or new lines.")
    return values


def parse_glucose_history(raw_text: str) -> list[float]:
    tokens = [token.strip() for token in raw_text.replace("\n", ",").split(",")]
    values = [float(token) for token in tokens if token]
    if len(values) != 23:
        raise ValueError("Enter exactly 23 previous glucose readings separated by commas or new lines.")
    return values


def compose_glucose_readings(readings: list[float], current_glucose: float) -> list[float]:
    if len(readings) == 24:
        history = readings[:-1]
    elif len(readings) == 23:
        history = readings
    else:
        raise ValueError("Expected either 23 previous readings or a 24-reading loaded trace.")
    return [float(value) for value in history] + [float(current_glucose)]


def _serialize_glucose_readings(readings: list[float]) -> str:
    return ", ".join(f"{float(value):.1f}" for value in readings)


def _recent_glucose_readings(report: dict[str, object]) -> list[float]:
    return [float(item["glucose"]) for item in report["recent_trace"][-24:]]


def _profile_seed(report: dict[str, object] | None) -> dict[str, object]:
    profile = {} if report is None else dict(report.get("profile") or {})
    return {
        "name": str(profile.get("name") or ""),
        "age": int(profile["age"]) if profile.get("age") is not None else 28,
        "diabetes_type": str(profile.get("diabetes_type") or "Type 1"),
        "insulin_therapy": str(profile.get("insulin_therapy") or "Bolus + basal"),
        "target_range_low": float(profile.get("target_range_low", 70.0)),
        "target_range_high": float(profile.get("target_range_high", 180.0)),
        "weight_kg": float(profile["weight_kg"]) if profile.get("weight_kg") is not None else 65.0,
    }


def _watch_preview_payload(report: dict[str, object] | None) -> dict[str, object]:
    if report is not None and report.get("watch"):
        return dict(report["watch"])
    if report is not None and {"risk", "trend", "reason"}.issubset(report.keys()):
        return dict(report)
    return {
        "glucose": None,
        "roc_15": None,
        "trend": "Waiting for live CGM feed",
        "risk": "UNKNOWN",
        "reason": "Connect real patient data to activate validated risk alerts.",
        "buzz": False,
        "forecast_30min": None,
        "forecast_warning": "",
        "hypo_probability": None,
        "top_reason": "Waiting for validated risk signals.",
        "watch_status": "Prediction unavailable",
        "updated_at": datetime.now().isoformat(),
        "status": "setup_required",
    }


def _friendly_top_factors(prediction: dict[str, object]) -> list[str]:
    lines: list[str] = []
    for factor in prediction.get("top_factors", [])[:3]:
        feature = factor["feature"]
        contribution = float(factor["contribution"])
        label = FRIENDLY_FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
        if contribution > 0:
            lines.append(f"{label} is increasing your low-glucose risk.")
        else:
            lines.append(f"{label} is lowering your risk right now.")
    return lines


def _render_daily_insights(prediction: dict[str, object]) -> None:
    st.markdown("### Why this matters")
    lines = _friendly_top_factors(prediction)
    if lines:
        for line in lines:
            st.write(f"- {line}")
    else:
        st.write("- The model is combining recent glucose, trend, insulin, and activity context for this prediction.")
    if prediction.get("forecast_notice"):
        st.info(str(prediction["forecast_notice"]))
    if prediction.get("status") != "ok":
        st.warning(str(prediction.get("abstention_reason")))
    st.caption("This tool is a decision-support demo and does not replace your care team plan.")


def _daily_guidance(prediction: dict[str, object]) -> dict[str, str]:
    if prediction.get("status", "ok") != "ok":
        return {
            "headline": "Prediction unavailable",
            "body": str(prediction.get("abstention_reason") or "The current input is outside the validated model range."),
        }
    if prediction.get("forecast_notice"):
        return {
            "headline": "Treat the current low glucose now",
            "body": str(prediction["forecast_notice"]),
        }
    risk = prediction["risk_level"]
    forecast_value = prediction.get("predicted_glucose_30min")
    if forecast_value is None:
        if risk == "HIGH":
            return {
                "headline": "High risk of hypoglycaemia soon",
                "body": "The model sees high near-term risk. Follow your hypo treatment plan and recheck soon.",
            }
        if risk == "MEDIUM":
            return {
                "headline": "Moderate risk, keep monitoring",
                "body": "The model sees moderate near-term risk. Keep monitoring symptoms and glucose closely.",
            }
        return {
            "headline": "Low immediate risk",
            "body": "No immediate alert is triggered.",
        }
    forecast = float(forecast_value)
    if risk == "HIGH":
        return {
            "headline": "High risk of hypoglycaemia soon",
            "body": f"Projected glucose is around {forecast:.0f} mg/dL in 30 minutes. Follow your hypo treatment plan and recheck soon.",
        }
    if risk == "MEDIUM":
        return {
            "headline": "Moderate risk, keep monitoring",
            "body": f"Projected glucose is around {forecast:.0f} mg/dL in 30 minutes. Keep an eye on symptoms and consider checking again soon.",
        }
    return {
        "headline": "Low immediate risk",
        "body": f"Projected glucose is around {forecast:.0f} mg/dL in 30 minutes. No immediate alert is triggered.",
    }


def _user_timeline_figure(readings: list[float], prediction: dict[str, object], timestamp: datetime | None = None) -> go.Figure:
    end = pd.Timestamp(timestamp or datetime.now()).floor("5min")
    recent = pd.DataFrame(
        {
            "timestamp": pd.date_range(end=end, periods=len(readings), freq="5min"),
            "glucose": readings,
        }
    )
    return _timeline_figure(recent, prediction)


def _forecast_only_figure(
    current_glucose: float | None,
    prediction: dict[str, object],
    timestamp: datetime | None = None,
) -> go.Figure:
    end = pd.Timestamp(timestamp or datetime.now()).floor("5min")
    fig = go.Figure()
    forecast_trace = prediction.get("forecast_trace") or []
    forecast_color = _chart_forecast_color(None if not forecast_trace else float(forecast_trace[-1]))
    interval_fill = _hex_to_rgba(forecast_color, 0.12)

    if current_glucose is not None:
        fig.add_trace(
            go.Scatter(
                x=[end],
                y=[float(current_glucose)],
                name="Current glucose",
                mode="markers",
                marker={
                    "color": _chart_glucose_color(current_glucose),
                    "size": 11,
                    "line": {"color": "#ffffff", "width": 1.5},
                },
            )
        )

    if forecast_trace:
        horizon_index = pd.date_range(
            start=end + pd.Timedelta(minutes=5),
            periods=len(forecast_trace),
            freq="5min",
        )
        fig.add_trace(
            go.Scatter(
                x=horizon_index,
                y=prediction["forecast_upper"],
                line={"color": "rgba(217,108,6,0)"},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=horizon_index,
                y=prediction["forecast_lower"],
                line={"color": "rgba(217,108,6,0)"},
                fill="tonexty",
                fillcolor=interval_fill,
                name="90% interval",
            )
        )
        path_x = list(horizon_index)
        path_y = [float(value) for value in forecast_trace]
        if current_glucose is not None:
            path_x = [end] + path_x
            path_y = [float(current_glucose)] + path_y
        fig.add_trace(
            go.Scatter(
                x=path_x,
                y=path_y,
                name="30-minute forecast",
                mode="lines+markers",
                line={"color": forecast_color, "width": 3},
                marker={"color": forecast_color, "size": 8, "line": {"color": "#ffffff", "width": 1}},
            )
        )

    fig.add_hline(y=70, line_dash="dot", line_color=PALETTE["high"])
    fig.add_hline(y=54, line_dash="dash", line_color="#7f1d1d")
    fig.update_layout(
        height=380,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.75)",
        legend={"orientation": "h"},
        xaxis_title="Forecast window",
        yaxis_title="Glucose (mg/dL)",
    )
    return fig


def _timeline_figure(recent: pd.DataFrame, prediction: dict[str, object]) -> go.Figure:
    recent = recent.copy()
    recent["timestamp"] = pd.to_datetime(recent["timestamp"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent["timestamp"],
            y=recent["glucose"],
            name="Recent glucose",
            mode="lines",
            line={"color": PALETTE["ink"], "width": 3},
        )
    )
    if prediction.get("forecast_trace"):
        horizon_index = pd.date_range(
            start=recent["timestamp"].iloc[-1] + pd.Timedelta(minutes=5),
            periods=len(prediction["forecast_trace"]),
            freq="5min",
        )
        fig.add_trace(
            go.Scatter(
                x=horizon_index,
                y=prediction["forecast_upper"],
                line={"color": "rgba(217,108,6,0)"},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=horizon_index,
                y=prediction["forecast_lower"],
                line={"color": "rgba(217,108,6,0)"},
                fill="tonexty",
                fillcolor="rgba(217,108,6,0.16)",
                name="90% interval",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=horizon_index,
                y=prediction["forecast_trace"],
                name="30-minute forecast",
                mode="lines+markers",
                line={"color": PALETTE["accent"], "width": 3},
            )
        )
    fig.add_hline(y=70, line_dash="dot", line_color=PALETTE["high"])
    fig.add_hline(y=54, line_dash="dash", line_color="#7f1d1d")
    fig.update_layout(
        height=380,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.75)",
        legend={"orientation": "h"},
        xaxis_title="Time",
        yaxis_title="Glucose (mg/dL)",
    )
    return fig


def _trace_figure(report: dict[str, object]) -> go.Figure:
    recent = pd.DataFrame(report["recent_trace"])
    prediction = report["prediction"]
    return _timeline_figure(recent, prediction)


def _research_controls(service) -> None:
    st.header("Demo Controls")
    ohio_root = st.text_input(
        "OhioT1DM root",
        value="data/raw/OhioT1DM",
        help="Folder containing OhioT1DM-training and OhioT1DM-testing.",
    )
    if st.button("Run Ohio Benchmark", width="stretch"):
        try:
            service.benchmark_ohio(ohio_root, persist=False, max_forecast_points=250)
            st.success("OhioT1DM benchmark completed.")
        except Exception as exc:  # pragma: no cover - Streamlit path
            st.error(str(exc))

    if st.button("Run Federated Demo", width="stretch"):
        try:
            service.run_federated_demo(rounds=3, min_clients=2)
            st.success("Federated demo completed.")
        except Exception as exc:  # pragma: no cover - Streamlit path
            st.error(str(exc))


def _render_watch_preview(report: dict[str, object] | None) -> None:
    payload = _watch_preview_payload(report)
    st.markdown(_watch_preview_html(payload), unsafe_allow_html=True)


def _watch_preview_html(payload: dict[str, object]) -> str:
    risk = str(payload.get("risk") or "UNKNOWN")
    risk_color = _risk_color(risk if risk != "UNKNOWN" else None)
    glucose = None if payload.get("glucose") is None else float(payload["glucose"])
    glucose_value = "--" if glucose is None else f"{glucose:.0f}"
    glucose_color = _glucose_value_color(glucose)
    roc_15 = float(payload.get("roc_15") or 0.0)
    trend_symbol = _trend_symbol(roc_15)
    trend_color = _trend_color(roc_15)
    trend_text = str(payload.get("trend") or "Trend unavailable")
    forecast = None if payload.get("forecast_30min") is None else float(payload["forecast_30min"])
    forecast_color = _forecast_value_color(forecast)
    forecast_value = "Forecast unavailable" if forecast is None else f"30 min: {forecast:.0f} mg/dL"
    forecast_warning = str(payload.get("forecast_warning") or "")
    reason = str(payload.get("top_reason") or payload.get("reason") or "Check dashboard")
    watch_status = str(payload.get("watch_status") or "Prediction unavailable")
    probability = payload.get("hypo_probability")
    probability_pct = 0.0 if probability in (None, "") else max(0.0, min(100.0, float(probability) * 100.0))
    probability_label = "unknown" if probability in (None, "") else f"{probability_pct:.0f}%"
    watch_color = "#94a3b8" if risk == "UNKNOWN" else risk_color
    risk_strip = "".join(
        (
            f"<div style='flex:1;padding:0.28rem 0.35rem;border-radius:999px;text-align:center;"
            f"background:{risk_color if level == risk else 'rgba(148,163,184,0.12)'};"
            f"color:{'#ffffff' if level == risk else '#94a3b8'};font-size:0.68rem;font-weight:700;'>{level}</div>"
        )
        for level in ("LOW", "MEDIUM", "HIGH")
    )
    forecast_detail = f"&#9888; {escape(forecast_warning)}" if forecast_warning else "Trajectory remains above the danger buffer."
    watch_icon = "&#9889;" if payload.get("buzz") else "&#9888;" if risk == "MEDIUM" else "&#10003;" if risk == "LOW" else "&#9711;"
    watch_detail = "High-risk alerts trigger the watch buzz." if payload.get("buzz") else "No watch buzz triggered right now."
    return dedent(
        f"""
        <div style="background:#081120;color:white;border-radius:28px;padding:1.2rem 1.2rem 1rem 1.2rem;min-height:360px;box-shadow:0 18px 40px rgba(0,0,0,0.18);display:flex;flex-direction:column;gap:1rem;">
            <div>
                <div style="display:inline-block;padding:0.35rem 0.75rem;border-radius:999px;background:{risk_color};font-size:0.78rem;font-weight:700;letter-spacing:0.08em;">{escape(risk)} RISK</div>
                <div style="display:flex;justify-content:space-between;align-items:flex-end;gap:1rem;margin-top:1rem;">
                    <div>
                        <div style="font-size:3.2rem;font-weight:800;line-height:0.95;color:{glucose_color};">{escape(glucose_value)}</div>
                        <div style="font-size:0.9rem;color:#94a3b8;margin-top:0.25rem;">mg/dL current glucose</div>
                    </div>
                    <div style="font-size:2rem;font-weight:800;color:{trend_color};line-height:1;">{escape(trend_symbol)}</div>
                </div>
                <div style="font-size:0.95rem;color:{trend_color};margin-top:0.4rem;font-weight:600;">{escape(trend_text)}</div>
            </div>
            <div style="display:flex;gap:0.35rem;">{risk_strip}</div>
            <div style="padding:0.8rem 0.9rem;border-radius:18px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.06);">
                <div style="font-size:0.82rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em;">30-minute forecast</div>
                <div style="font-size:1.25rem;font-weight:800;color:{forecast_color};margin-top:0.25rem;">{escape(forecast_value)}</div>
                <div style="font-size:0.82rem;color:{forecast_color};margin-top:0.35rem;">{forecast_detail}</div>
            </div>
            <div style="padding:0.75rem 0.9rem;border-radius:18px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.06);">
                <div style="font-size:0.82rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em;">Top reason</div>
                <div style="font-size:0.98rem;color:#e2e8f0;margin-top:0.35rem;line-height:1.45;">{escape(reason)}</div>
            </div>
            <div>
                <div style="display:flex;justify-content:space-between;align-items:center;font-size:0.82rem;color:#94a3b8;">
                    <span>Hypo probability</span>
                    <span>{escape(probability_label)}</span>
                </div>
                <div style="height:8px;background:rgba(148,163,184,0.18);border-radius:999px;overflow:hidden;margin-top:0.35rem;">
                    <div style="width:{probability_pct:.1f}%;height:100%;background:{risk_color};"></div>
                </div>
            </div>
            <div style="padding:0.8rem 0.9rem;border-radius:18px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.06);">
                <div style="font-size:0.9rem;font-weight:700;color:{watch_color};">{watch_icon} {escape(watch_status)}</div>
                <div style="font-size:0.75rem;color:#8ea0b8;margin-top:0.45rem;">{escape(watch_detail)}</div>
            </div>
        </div>
        """
    ).strip()


def _watch_reason_from_prediction(prediction: dict[str, object]) -> str:
    if prediction.get("status") != "ok":
        return str(prediction.get("abstention_reason") or "Prediction unavailable")
    if prediction.get("top_reason"):
        return str(prediction["top_reason"])
    if prediction.get("forecast_notice"):
        return "Current glucose is already very low"
    top_factors = prediction.get("top_factors") or []
    if top_factors:
        message = str(top_factors[0].get("message") or "")
        return message.replace("Increases risk: ", "").replace("Reduces risk: ", "")[:52] or "Monitor glucose closely"
    explanation = str(prediction.get("explanation") or "").strip()
    return explanation[:52] if explanation else "Monitor glucose closely"


def _active_watch_payload(
    readings: list[float],
    prediction: dict[str, object],
    timestamp: datetime | None = None,
) -> dict[str, object]:
    current_glucose = float(readings[-1]) if readings else None
    roc_15 = float(readings[-1] - readings[-4]) if len(readings) >= 4 else 0.0
    risk = prediction.get("risk_level") or "UNKNOWN"
    return {
        "glucose": current_glucose,
        "roc_15": roc_15,
        "trend": f"{roc_15:+.1f} mg/dL per 15min",
        "risk": risk,
        "reason": _watch_reason_from_prediction(prediction),
        "buzz": bool(prediction.get("watch_buzz", prediction.get("alert_required"))),
        "forecast_30min": prediction.get("predicted_glucose_30min"),
        "forecast_warning": str(prediction.get("forecast_warning") or ""),
        "hypo_probability": prediction.get("hypo_probability"),
        "top_reason": str(prediction.get("top_reason") or _watch_reason_from_prediction(prediction)),
        "watch_status": str(prediction.get("watch_status") or "Prediction unavailable"),
        "updated_at": (timestamp or datetime.now()).isoformat(),
        "status": prediction.get("status", "live"),
    }


def _render_onboarding_mode(service) -> None:
    profile_state = st.session_state.setdefault("profile_draft", _profile_seed(None))
    meal_log = st.session_state.setdefault("local_meal_log", [])
    insulin_log = st.session_state.setdefault("local_insulin_log", [])

    with st.sidebar:
        st.header("Patient App")
        if st.button("Switch to Research Demo", width="stretch"):
            st.session_state["dashboard_mode"] = "Research Demo"
            st.rerun()
        st.caption("Clinical scoring stays locked until validated real patient data is loaded.")
        _research_controls(service)

    st.markdown(
        """
        <div class="hero-card">
            <div style="font-size:0.95rem;color:#5b6770;text-transform:uppercase;letter-spacing:0.08em;">Patient App Mode</div>
            <div style="font-size:2.1rem;font-weight:700;color:#15222e;">Set up once. Log meals and insulin. Everything else stays automatic.</div>
            <div style="font-size:1rem;color:#5b6770;max-width:860px;">
                GlycoGuard is designed around a simple daily flow: CGM, activity, sleep, and time-of-day track silently in the background.
                The patient only logs meals and insulin, and the system runs a 30-minute hypoglycaemia check every 5 minutes once real data is connected.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    setup_col, auto_col, watch_col = st.columns([1.25, 1.0, 1.0])
    with setup_col:
        st.markdown("### Setup once")
        with st.form("onboarding-profile-form"):
            name = st.text_input("Name", value=str(profile_state["name"]))
            age = st.number_input("Age", min_value=1, max_value=120, value=int(profile_state["age"]), step=1)
            diabetes_type = st.selectbox("Diabetes type", ["Type 1", "Type 2", "Insulin-treated diabetes"], index=["Type 1", "Type 2", "Insulin-treated diabetes"].index(str(profile_state["diabetes_type"])) if str(profile_state["diabetes_type"]) in {"Type 1", "Type 2", "Insulin-treated diabetes"} else 0)
            insulin_therapy = st.selectbox("Insulin therapy", ["Bolus + basal", "Pump", "Fast-acting only", "Long-acting only"], index=["Bolus + basal", "Pump", "Fast-acting only", "Long-acting only"].index(str(profile_state["insulin_therapy"])) if str(profile_state["insulin_therapy"]) in {"Bolus + basal", "Pump", "Fast-acting only", "Long-acting only"} else 0)
            range_col1, range_col2 = st.columns(2)
            target_low = range_col1.number_input("Target low", min_value=54.0, max_value=120.0, value=float(profile_state["target_range_low"]), step=1.0)
            target_high = range_col2.number_input("Target high", min_value=120.0, max_value=250.0, value=float(profile_state["target_range_high"]), step=1.0)
            weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=float(profile_state["weight_kg"]), step=0.5)
            if st.form_submit_button("Save setup", width="stretch"):
                st.session_state["profile_draft"] = {
                    "name": name,
                    "age": int(age),
                    "diabetes_type": diabetes_type,
                    "insulin_therapy": insulin_therapy,
                    "target_range_low": float(target_low),
                    "target_range_high": float(target_high),
                    "weight_kg": float(weight),
                }
                st.success("Setup saved for this session.")

    with auto_col:
        st.markdown("### Automatic tracking")
        st.markdown(
            """
            <div class="metric-chip"><strong>CGM</strong><br/>Live 5-minute glucose feed when connected.</div>
            <div style="height:0.7rem;"></div>
            <div class="metric-chip"><strong>Phone / Watch</strong><br/>Activity, heart rate, sleep, and time-of-day arrive automatically.</div>
            <div style="height:0.7rem;"></div>
            <div class="metric-chip"><strong>Background model</strong><br/>Runs silently every 5 minutes and alerts only on meaningful risk.</div>
            """,
            unsafe_allow_html=True,
        )

    with watch_col:
        st.markdown("### Minimal watch alert")
        _render_watch_preview(None)

    log_col, next_col = st.columns([1.2, 1.0])
    with log_col:
        st.markdown("### The only two daily actions")
        meal_form, insulin_form = st.columns(2)
        with meal_form:
            with st.form("onboarding-meal-form"):
                meal_desc = st.text_input("Meal", value="Rice and dal")
                meal_carbs = st.number_input("Carbs (g)", min_value=0.0, max_value=150.0, value=45.0, step=1.0)
                if st.form_submit_button("Log meal", width="stretch"):
                    meal_log.append({"meal": meal_desc, "carbs": float(meal_carbs), "time": datetime.now().strftime("%H:%M")})
                    st.success("Meal saved locally. Live risk scoring unlocks after real data is connected.")
        with insulin_form:
            with st.form("onboarding-insulin-form"):
                insulin_units = st.number_input("Units", min_value=0.0, max_value=25.0, value=4.0, step=0.5)
                insulin_type = st.selectbox("Type", ["bolus", "basal"])
                if st.form_submit_button("Log insulin", width="stretch"):
                    insulin_log.append({"units": float(insulin_units), "type": insulin_type, "time": datetime.now().strftime("%H:%M")})
                    st.success("Insulin saved locally. Live risk scoring unlocks after real data is connected.")

    with next_col:
        st.markdown("### Activation status")
        st.info("Validated prediction is currently locked because strict mode requires real patient data and calibrated artifacts.")
        st.code("python -m glycoguard.cli benchmark-ohio data\\raw\\OhioT1DM --max-forecast-points 250")
        st.caption("This keeps the live product honest: no synthetic guesses, no silent fallback model.")

    history_col1, history_col2 = st.columns(2)
    with history_col1:
        st.markdown("#### Recent meals")
        if meal_log:
            st.dataframe(pd.DataFrame(meal_log[::-1]), width="stretch", hide_index=True)
        else:
            st.caption("No meals logged yet.")
    with history_col2:
        st.markdown("#### Recent insulin logs")
        if insulin_log:
            st.dataframe(pd.DataFrame(insulin_log[::-1]), width="stretch", hide_index=True)
        else:
            st.caption("No insulin logs yet.")


def _render_daily_user_mode(service, report: dict[str, object]) -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div style="font-size:0.95rem;color:#5b6770;text-transform:uppercase;letter-spacing:0.08em;">Daily Mode</div>
            <div style="font-size:2rem;font-weight:700;color:#15222e;">Simple Hypoglycaemia Check</div>
            <div style="font-size:1rem;color:#5b6770;max-width:760px;">
                Use the currently loaded patient or paste your last 24 glucose readings for a quick risk check.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    default_readings = _recent_glucose_readings(report)
    default_context = report["context"]
    profile_seed = st.session_state.setdefault("profile_draft", _profile_seed(report))

    with st.sidebar:
        st.header("Patient App")
        with st.form("daily-profile-form"):
            st.caption("Setup once")
            profile_name = st.text_input("Name", value=str(profile_seed["name"]))
            profile_age = st.number_input("Age", min_value=1, max_value=120, value=int(profile_seed["age"]), step=1)
            profile_diabetes = st.selectbox("Diabetes type", ["Type 1", "Type 2", "Insulin-treated diabetes"], index=["Type 1", "Type 2", "Insulin-treated diabetes"].index(str(profile_seed["diabetes_type"])) if str(profile_seed["diabetes_type"]) in {"Type 1", "Type 2", "Insulin-treated diabetes"} else 0)
            profile_therapy = st.selectbox("Insulin therapy", ["Bolus + basal", "Pump", "Fast-acting only", "Long-acting only"], index=["Bolus + basal", "Pump", "Fast-acting only", "Long-acting only"].index(str(profile_seed["insulin_therapy"])) if str(profile_seed["insulin_therapy"]) in {"Bolus + basal", "Pump", "Fast-acting only", "Long-acting only"} else 0)
            pr1, pr2 = st.columns(2)
            profile_low = pr1.number_input("Target low", min_value=54.0, max_value=120.0, value=float(profile_seed["target_range_low"]), step=1.0)
            profile_high = pr2.number_input("Target high", min_value=120.0, max_value=250.0, value=float(profile_seed["target_range_high"]), step=1.0)
            profile_weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=float(profile_seed["weight_kg"]), step=0.5)
            if st.form_submit_button("Save setup", width="stretch"):
                st.session_state["profile_draft"] = {
                    "name": profile_name,
                    "age": int(profile_age),
                    "diabetes_type": profile_diabetes,
                    "insulin_therapy": profile_therapy,
                    "target_range_low": float(profile_low),
                    "target_range_high": float(profile_high),
                    "weight_kg": float(profile_weight),
                }
                service.update_patient_profile(
                    report["patient_id"],
                    name=profile_name,
                    age=int(profile_age),
                    diabetes_type=profile_diabetes,
                    insulin_therapy=profile_therapy,
                    target_range_low=float(profile_low),
                    target_range_high=float(profile_high),
                    weight_kg=float(profile_weight),
                )
                st.success("Profile updated.")
                st.rerun()

        st.divider()
        st.subheader("Quick Check")
        input_source = st.radio("Input source", ["Use loaded patient", "Enter my own readings"], width="stretch")
        with st.form("daily-user-form"):
            raw_text = ""
            if input_source == "Enter my own readings":
                raw_text = st.text_area(
                    "Previous 23 glucose readings",
                    value=st.session_state.get("daily_manual_history", _serialize_glucose_readings(default_readings[:-1])),
                    help="Paste the 23 readings before the current one, separated by commas or new lines.",
                    height=120,
                )
                current_glucose = st.number_input(
                    "Current glucose",
                    min_value=40.0,
                    max_value=400.0,
                    value=float(st.session_state.get("daily_manual_current_glucose", default_readings[-1])),
                    step=1.0,
                )
            else:
                st.caption("Using the loaded patient's previous 23 CGM readings. You can override the current glucose below.")
                current_glucose = st.number_input(
                    "Current glucose",
                    min_value=40.0,
                    max_value=400.0,
                    value=float(st.session_state.get("daily_loaded_current_glucose", default_readings[-1])),
                    step=1.0,
                )

            carbs_last_hour = st.number_input(
                "Carbs in last hour (g)",
                min_value=0.0,
                max_value=150.0,
                value=float(default_context["carbs_1h"]),
                step=1.0,
            )
            insulin_on_board = st.number_input(
                "Active insulin",
                min_value=0.0,
                max_value=25.0,
                value=float(default_context["insulin_on_board"]),
                step=0.1,
            )
            activity_level = st.slider("Recent activity", min_value=0.0, max_value=1.0, value=float(default_context["activity"]), step=0.05)
            sleep_flag = st.checkbox("Sleeping or overnight period", value=bool(default_context["sleep_flag"]))
            stress_score = st.slider("Stress level", min_value=0.0, max_value=1.0, value=float(default_context["stress_score"]), step=0.05)
            submitted = st.form_submit_button("Check my risk", width="stretch")

        if st.button("Switch to Research Demo", width="stretch"):
            st.session_state["dashboard_mode"] = "Research Demo"
            st.rerun()

        st.divider()
        meal_col, insulin_col = st.columns(2)
        with meal_col:
            with st.form("daily-meal-form"):
                st.caption("Log meal")
                meal_desc = st.text_input("Meal", value="Lunch")
                meal_carbs = st.number_input("Carbs", min_value=0.0, max_value=150.0, value=45.0, step=1.0)
                if st.form_submit_button("Save meal", width="stretch"):
                    service.log_meal(report["patient_id"], carb_grams=float(meal_carbs), description=meal_desc)
                    st.success("Meal logged.")
                    st.rerun()
        with insulin_col:
            with st.form("daily-insulin-form"):
                st.caption("Log insulin")
                insulin_units = st.number_input("Units", min_value=0.0, max_value=25.0, value=4.0, step=0.5)
                insulin_type = st.selectbox("Type", ["bolus", "basal"])
                if st.form_submit_button("Save insulin", width="stretch"):
                    service.log_insulin(report["patient_id"], insulin_units=float(insulin_units), insulin_type=insulin_type)
                    st.success("Insulin logged.")
                    st.rerun()

    active_prediction = report["prediction"]
    active_readings = default_readings
    input_caption = f"Using patient {report['patient_id']}."

    if submitted:
        try:
            if input_source == "Use loaded patient":
                readings = compose_glucose_readings(default_readings, current_glucose)
                patient_id = report["patient_id"]
                st.session_state["daily_loaded_current_glucose"] = float(current_glucose)
            else:
                readings = compose_glucose_readings(parse_glucose_history(raw_text), current_glucose)
                patient_id = "manual-user"
                st.session_state["daily_manual_history"] = raw_text
                st.session_state["daily_manual_current_glucose"] = float(current_glucose)

            payload = CGMInput(
                patient_id=patient_id,
                glucose_readings=readings,
                carbs_last_hour=float(carbs_last_hour),
                insulin_on_board=float(insulin_on_board),
                activity_level=float(activity_level),
                sleep_flag=int(sleep_flag),
                stress_score=float(stress_score),
            )
            active_prediction = service.explain(payload)
            active_readings = readings
            st.session_state["daily_prediction"] = active_prediction
            st.session_state["daily_readings"] = readings
            st.session_state["daily_source"] = input_source
            input_caption = (
                "Using your manual readings."
                if input_source == "Enter my own readings"
                else f"Using patient {report['patient_id']} history with current glucose {float(current_glucose):.1f} mg/dL."
            )
        except ValueError as exc:
            st.error(str(exc))
            stored_prediction = st.session_state.get("daily_prediction")
            stored_readings = st.session_state.get("daily_readings")
            if stored_prediction and stored_readings:
                active_prediction = stored_prediction
                active_readings = stored_readings
                input_caption = "Showing your last successful manual check."
    elif input_source == "Enter my own readings":
        stored_prediction = st.session_state.get("daily_prediction")
        stored_readings = st.session_state.get("daily_readings")
        if stored_prediction and stored_readings and st.session_state.get("daily_source") == "Enter my own readings":
            active_prediction = stored_prediction
            active_readings = stored_readings
            input_caption = "Showing your last successful manual check."
        else:
            st.info("Enter the previous 23 readings plus the current glucose, then press 'Check my risk'.")

    guidance = _daily_guidance(active_prediction)
    risk_color = _risk_color(active_prediction["risk_level"])
    active_watch = _active_watch_payload(active_readings, active_prediction)

    st.caption(input_caption)
    st.markdown(
        f"""
        <div style="padding:1rem 1.2rem;border-radius:18px;background:{risk_color};color:white;margin-top:0.4rem;margin-bottom:1rem;">
            <div style="font-size:0.9rem;opacity:0.9;">{active_prediction["risk_level"]} RISK</div>
            <div style="font-size:1.5rem;font-weight:700;">{guidance["headline"]}</div>
            <div style="font-size:1rem;opacity:0.95;">{guidance["body"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Current glucose", f"{active_readings[-1]:.1f} mg/dL")
    col2.metric(
        "Risk score",
        "unknown" if active_prediction.get("hypo_probability") is None else f'{active_prediction["hypo_probability"] * 100:.1f}%',
    )
    col3.metric(
        "30-min forecast",
        "unknown" if active_prediction.get("predicted_glucose_30min") is None else f'{active_prediction["predicted_glucose_30min"]} mg/dL',
    )

    top_left, top_right = st.columns([2.0, 1.0])
    with top_left:
        st.plotly_chart(_forecast_only_figure(active_readings[-1], active_prediction), width="stretch")
    with top_right:
        _render_watch_preview(active_watch)

    bottom_left, bottom_right = st.columns([1.0, 1.2])
    with bottom_left:
        st.plotly_chart(
            _gauge_figure(float(active_prediction.get("hypo_probability") or 0.0), active_prediction.get("risk_level")),
            width="stretch",
        )
    with bottom_right:
        _render_daily_insights(active_prediction)

    with st.expander("See advanced model details"):
        waterfall_payload = active_prediction.get("waterfall") or report.get("waterfall", {})
        rendered = _render_waterfall(waterfall_payload)
        if not rendered:
            st.plotly_chart(_explanation_figure(active_prediction), width="stretch")
        st.write(
            {
                "carbs_last_hour": float(carbs_last_hour),
                "insulin_on_board": float(insulin_on_board),
                "activity_level": float(activity_level),
                "sleep_flag": int(sleep_flag),
                "stress_score": float(stress_score),
            }
        )


def _render_research_demo_mode(service, report: dict[str, object]) -> None:
    with st.sidebar:
        _research_controls(service)
        if st.button("Switch to Patient App", width="stretch"):
            st.session_state["dashboard_mode"] = "Patient App"
            st.rerun()


def _gauge_figure(probability: float, risk_level: str) -> go.Figure:
    color = _risk_color(risk_level)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100.0,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "rgba(15,118,110,0.16)"},
                    {"range": [40, 70], "color": "rgba(217,119,6,0.18)"},
                    {"range": [70, 100], "color": "rgba(180,35,24,0.18)"},
                ],
            },
            title={"text": "Hypoglycaemia risk"},
        )
    )
    fig.update_layout(height=280, margin={"l": 10, "r": 10, "t": 40, "b": 10}, paper_bgcolor="rgba(0,0,0,0)")
    return fig


def _explanation_figure(prediction: dict[str, object]) -> go.Figure:
    factors = pd.DataFrame(prediction["top_factors"])
    factors = factors.iloc[::-1]
    colors = [PALETTE["high"] if value >= 0 else PALETTE["low"] for value in factors["contribution"]]
    fig = go.Figure(
        go.Bar(
            x=factors["contribution"],
            y=factors["feature"],
            orientation="h",
            marker_color=colors,
            text=[message.replace("Increases risk: ", "").replace("Reduces risk: ", "") for message in factors["message"]],
            hovertext=factors["message"],
        )
    )
    fig.update_layout(
        height=320,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.75)",
        xaxis_title="Contribution",
        yaxis_title="Feature",
    )
    return fig


def _render_waterfall(waterfall: dict[str, object]) -> bool:
    if shap is None or waterfall.get("backend") != "shap":
        return False

    feature_names = waterfall["feature_names"]
    shap_values = np.asarray([waterfall["shap_values"][name] for name in feature_names], dtype=float)
    feature_values = np.asarray([waterfall["feature_values"][name] for name in feature_names], dtype=float)
    explanation = shap.Explanation(
        values=shap_values,
        base_values=float(waterfall["base_value"]),
        data=feature_values,
        feature_names=feature_names,
    )
    fig = plt.figure(figsize=(8, 5))
    shap.plots.waterfall(explanation, max_display=8, show=False)
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    return True


def _agp_figure(report: dict[str, object]) -> go.Figure:
    agp = pd.DataFrame(report["agp"]["percentiles"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=agp["time"],
            y=agp["p95"],
            line={"color": "rgba(217,108,6,0)"},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=agp["time"],
            y=agp["p05"],
            line={"color": "rgba(217,108,6,0)"},
            fill="tonexty",
            fillcolor="rgba(217,108,6,0.12)",
            name="5th-95th percentile",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=agp["time"],
            y=agp["p75"],
            line={"color": "rgba(21,34,46,0)"},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=agp["time"],
            y=agp["p25"],
            line={"color": "rgba(21,34,46,0)"},
            fill="tonexty",
            fillcolor="rgba(21,34,46,0.16)",
            name="25th-75th percentile",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=agp["time"],
            y=agp["p50"],
            mode="lines",
            line={"color": PALETTE["ink"], "width": 2.5},
            name="Median",
        )
    )
    fig.add_hline(y=70, line_dash="dot", line_color=PALETTE["high"])
    fig.add_hline(y=180, line_dash="dot", line_color=PALETTE["accent"])
    fig.update_layout(
        height=360,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.75)",
        xaxis_title="Time of day",
        yaxis_title="Glucose (mg/dL)",
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="GlycoGuard", layout="wide", page_icon="GG")
    _inject_css()

    service = get_service()
    if "dashboard_mode" not in st.session_state:
        st.session_state["dashboard_mode"] = "Patient App"

    if st.session_state["dashboard_mode"] == "Daily User":
        st.session_state["dashboard_mode"] = "Patient App"

    if st.session_state["dashboard_mode"] == "Patient App":
        if not service.is_ready() or not service.records:
            _render_onboarding_mode(service)
            return
        report = service.get_report()
        _render_daily_user_mode(service, report)
        return

    if not service.is_ready() or not service.records:
        st.markdown(
            """
            <div class="hero-card">
                <div style="font-size:0.95rem;color:#5b6770;text-transform:uppercase;letter-spacing:0.08em;">Research Demo</div>
                <div style="font-size:2rem;font-weight:700;color:#15222e;">Real data required before benchmark and report views</div>
                <div style="font-size:1rem;color:#5b6770;max-width:760px;">
                    Load OhioT1DM or a real patient bundle to activate the strict calibrated dashboard. The patient app mode remains available for product walkthrough.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.sidebar:
            _research_controls(service)
        return

    report = service.get_report()
    _render_research_demo_mode(service, report)
    prediction = report["prediction"]
    risk_color = _risk_color(prediction["risk_level"])

    st.markdown(
        f"""
        <div class="hero-card">
            <div style="display:flex;justify-content:space-between;gap:1rem;align-items:flex-start;">
                <div>
                    <div style="font-size:0.95rem;color:{PALETTE["muted"]};text-transform:uppercase;letter-spacing:0.08em;">GlycoGuard</div>
                    <div style="font-size:2.2rem;font-weight:700;color:{PALETTE["ink"]};">Context-Aware Hypoglycaemia Risk Intelligence</div>
                    <div style="font-size:1rem;color:{PALETTE["muted"]};max-width:800px;">
                        Loaded patient {report["patient_id"]} running through the strict calibrated production pipeline.
                    </div>
                </div>
                <div style="padding:0.5rem 1rem;border-radius:999px;background:{risk_color};color:white;font-weight:700;">
                    {prediction["risk_level"]} RISK
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current glucose", f'{report["current_glucose"]} mg/dL')
    col2.metric("15-min delta", f'{report["roc_15"]} mg/dL')
    col3.metric("Risk score", f'{prediction["hypo_probability"] * 100:.1f}%')
    col4.metric("30-min forecast", f'{prediction["predicted_glucose_30min"]} mg/dL')

    left, right = st.columns([2.2, 1.0])
    with left:
        st.plotly_chart(_trace_figure(report), width="stretch")
    with right:
        st.plotly_chart(_gauge_figure(prediction["hypo_probability"], prediction["risk_level"]), width="stretch")
        st.markdown("### Why the alert fired")
        st.caption(prediction["explanation"])
        st.write(
            {
                "carbs_1h": report["context"]["carbs_1h"],
                "insulin_on_board": report["context"]["insulin_on_board"],
                "activity": report["context"]["activity"],
                "sleep_flag": report["context"]["sleep_flag"],
                "stress_score": report["context"]["stress_score"],
            }
        )

    exp_col, agp_col = st.columns([1.0, 1.3])
    with exp_col:
        st.subheader("SHAP Waterfall")
        rendered = _render_waterfall(report["waterfall"])
        if not rendered:
            st.caption("SHAP waterfall unavailable for the active backend; showing contribution view instead.")
            st.plotly_chart(_explanation_figure(prediction), width="stretch")
    with agp_col:
        st.subheader("Ambulatory Glucose Profile")
        st.plotly_chart(_agp_figure(report), width="stretch")

    metrics_col, log_col = st.columns([1.0, 1.2])
    with metrics_col:
        st.subheader("Clinical metrics")
        metrics = report["metrics"]
        st.write(
            {
                "AUC-ROC": round(metrics["auc_roc"], 3),
                "Sensitivity": round(metrics["sensitivity"], 3),
                "Specificity": round(metrics["specificity"], 3),
                "F1": round(metrics["f1"], 3),
                "Alert rate": round(metrics["alert_rate"], 3),
            }
        )
        st.subheader("AGP summary")
        st.write(report["agp"]["summary"])

    with log_col:
        st.subheader("Recent alert log")
        st.dataframe(pd.DataFrame(report["alert_log"]), width="stretch", hide_index=True)

    benchmark = report.get("benchmark")
    if benchmark:
        st.subheader("OhioT1DM Benchmark")
        bcol1, bcol2, bcol3 = st.columns(3)
        bcol1.metric("AUC-ROC", f'{benchmark["overall_metrics"].get("auc_roc", 0.0):.3f}')
        bcol2.metric("Lead time", f'{benchmark["lead_time"].get("mean_minutes", 0.0):.1f} min')
        bcol3.metric("Clarke A+B", f'{benchmark["clarke_grid"].get("zone_ab", 0.0) * 100:.1f}%')
        st.dataframe(pd.DataFrame(benchmark["per_patient"]), width="stretch", hide_index=True)

    federated = report.get("federated")
    if federated:
        st.subheader("Federated Learning Demo")
        fcol1, fcol2, fcol3 = st.columns(3)
        fcol1.metric("Clients", str(federated.get("num_clients", 0)))
        fcol2.metric("Rounds", str(federated.get("rounds", 0)))
        fcol3.metric("Federated AUC", f'{federated.get("federated_auc", 0.0):.3f}')
        st.dataframe(pd.DataFrame(federated.get("round_metrics", [])), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
