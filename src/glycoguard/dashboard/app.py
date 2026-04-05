from __future__ import annotations

from datetime import datetime
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
    "high": "#b42318",
    "medium": "#d97706",
    "low": "#0f766e",
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


def parse_glucose_readings(raw_text: str) -> list[float]:
    tokens = [token.strip() for token in raw_text.replace("\n", ",").split(",")]
    values = [float(token) for token in tokens if token]
    if len(values) != 24:
        raise ValueError("Enter exactly 24 glucose readings separated by commas or new lines.")
    return values


def _serialize_glucose_readings(readings: list[float]) -> str:
    return ", ".join(f"{float(value):.1f}" for value in readings)


def _recent_glucose_readings(report: dict[str, object]) -> list[float]:
    return [float(item["glucose"]) for item in report["recent_trace"][-24:]]


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


def _daily_guidance(prediction: dict[str, object]) -> dict[str, str]:
    if prediction.get("status", "ok") != "ok":
        return {
            "headline": "Prediction unavailable",
            "body": str(prediction.get("abstention_reason") or "The current input is outside the validated model range."),
        }
    risk = prediction["risk_level"]
    forecast = float(prediction["predicted_glucose_30min"])
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


def _timeline_figure(recent: pd.DataFrame, prediction: dict[str, object]) -> go.Figure:
    recent = recent.copy()
    recent["timestamp"] = pd.to_datetime(recent["timestamp"])
    horizon_index = pd.date_range(
        start=recent["timestamp"].iloc[-1] + pd.Timedelta(minutes=5),
        periods=len(prediction["forecast_trace"]),
        freq="5min",
    )
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

    with st.sidebar:
        st.header("Quick Check")
        input_source = st.radio("Input source", ["Use loaded patient", "Enter my own readings"], width="stretch")
        with st.form("daily-user-form"):
            raw_text = ""
            if input_source == "Enter my own readings":
                raw_text = st.text_area(
                    "Last 24 glucose readings",
                    value=st.session_state.get("daily_manual_readings", _serialize_glucose_readings(default_readings)),
                    help="Paste 24 values separated by commas or new lines.",
                    height=120,
                )

            carbs_last_hour = st.number_input("Carbs in last hour (g)", min_value=0.0, value=float(default_context["carbs_1h"]), step=1.0)
            insulin_on_board = st.number_input("Active insulin", min_value=0.0, value=float(default_context["insulin_on_board"]), step=0.1)
            activity_level = st.slider("Recent activity", min_value=0.0, max_value=1.0, value=float(default_context["activity"]), step=0.05)
            sleep_flag = st.checkbox("Sleeping or overnight period", value=bool(default_context["sleep_flag"]))
            stress_score = st.slider("Stress level", min_value=0.0, max_value=1.0, value=float(default_context["stress_score"]), step=0.05)
            submitted = st.form_submit_button("Check my risk", width="stretch")

        if st.button("Switch to Research Demo", width="stretch"):
            st.session_state["dashboard_mode"] = "Research Demo"
            st.rerun()

    active_prediction = report["prediction"]
    active_readings = default_readings
    input_caption = f"Using patient {report['patient_id']}."

    if submitted:
        try:
            if input_source == "Use loaded patient":
                readings = default_readings
                patient_id = report["patient_id"]
            else:
                readings = parse_glucose_readings(raw_text)
                patient_id = "manual-user"
                st.session_state["daily_manual_readings"] = raw_text

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
            input_caption = "Using your manual readings." if input_source == "Enter my own readings" else f"Using patient {report['patient_id']}."
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
            st.info("Paste 24 readings and press 'Check my risk' to run a personal check.")

    guidance = _daily_guidance(active_prediction)
    risk_color = _risk_color(active_prediction["risk_level"])

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

    left, right = st.columns([2.0, 1.0])
    with left:
        st.plotly_chart(_user_timeline_figure(active_readings, active_prediction), width="stretch")
    with right:
        st.plotly_chart(
            _gauge_figure(float(active_prediction.get("hypo_probability") or 0.0), active_prediction.get("risk_level")),
            width="stretch",
        )
        st.markdown("### Why this matters")
        for line in _friendly_top_factors(active_prediction):
            st.write(f"- {line}")
        if active_prediction.get("status") != "ok":
            st.warning(str(active_prediction.get("abstention_reason")))
        st.caption("This tool is a decision-support demo and does not replace your care team plan.")

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
        if st.button("Switch to Daily Mode", width="stretch"):
            st.session_state["dashboard_mode"] = "Daily User"
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
    if not service.is_ready() or not service.records:
        st.markdown(
            """
            <div class="hero-card">
                <div style="font-size:0.95rem;color:#5b6770;text-transform:uppercase;letter-spacing:0.08em;">Strict Production Mode</div>
                <div style="font-size:2rem;font-weight:700;color:#15222e;">Real data required before prediction</div>
                <div style="font-size:1rem;color:#5b6770;max-width:760px;">
                    Synthetic startup has been removed. Ingest OhioT1DM or a real patient bundle, then retrain the calibrated model before opening the dashboard.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.sidebar:
            _research_controls(service)
        st.code("python -m glycoguard.cli benchmark-ohio data\\raw\\OhioT1DM --max-forecast-points 250")
        return

    report = service.get_report()
    if "dashboard_mode" not in st.session_state:
        st.session_state["dashboard_mode"] = "Daily User"

    if st.session_state["dashboard_mode"] == "Daily User":
        _render_daily_user_mode(service, report)
        return

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
