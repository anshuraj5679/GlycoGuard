from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse

from glycoguard.federated.client import federated_status
from glycoguard.config import load_config
from glycoguard.gemini_audit import run_prediction_audit
from glycoguard.schemas import (
    AuditRequest,
    ArtifactRequest,
    BundleIngestRequest,
    CGMInput,
    ExplanationResponse,
    FederatedRunRequest,
    HealthResponse,
    InsulinLogRequest,
    MealLogRequest,
    OhioBenchmarkRequest,
    OhioIngestRequest,
    PatientProfile,
    PatientProfileUpdateRequest,
    PredictionResponse,
    ReplayStartRequest,
    ReplayStepResponse,
    TrainRequest,
    WatchPayloadResponse,
)
from glycoguard.service import get_service


def create_app() -> FastAPI:
    config = load_config("configs/default.yaml")
    app = FastAPI(title=config.api.title, version=config.api.version)

    @app.get("/health", response_model=HealthResponse)
    def health() -> dict[str, object]:
        service = get_service()
        return {
            "status": "ok",
            "ready": service.is_ready(),
            "api_version": config.api.version,
            "model_backend": None if service.model_bundle is None else service.model_bundle.backend,
            "forecast_backend": None if service.forecaster is None else service.forecaster.backend,
            "default_patient_id": service.default_patient_id,
            "strict_mode": config.model.strict_mode,
            "real_patient_count": len(service.records),
        }

    @app.get("/patients")
    def patients() -> list[dict[str, object]]:
        return get_service().list_patients()

    @app.get("/profile/{patient_id}", response_model=PatientProfile)
    def patient_profile(patient_id: str) -> dict[str, object]:
        service = get_service()
        try:
            return service.get_patient_profile(patient_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/profile/{patient_id}", response_model=PatientProfile)
    def update_patient_profile(patient_id: str, payload: PatientProfileUpdateRequest) -> dict[str, object]:
        service = get_service()
        try:
            return service.update_patient_profile(patient_id, **payload.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/log/meal")
    def log_meal(payload: MealLogRequest) -> dict[str, object]:
        service = get_service()
        try:
            return service.log_meal(
                patient_id=payload.patient_id,
                carb_grams=payload.carb_grams,
                timestamp=payload.timestamp,
                description=payload.description,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/log/insulin")
    def log_insulin(payload: InsulinLogRequest) -> dict[str, object]:
        service = get_service()
        try:
            return service.log_insulin(
                patient_id=payload.patient_id,
                insulin_units=payload.insulin_units,
                timestamp=payload.timestamp,
                insulin_type=payload.insulin_type,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/predict", response_model=PredictionResponse)
    def predict(data: CGMInput) -> dict[str, object]:
        try:
            service = get_service()
            result = service.predict(data)
            result.pop("feature_frame", None)
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/explain", response_model=ExplanationResponse)
    def explain(data: CGMInput) -> dict[str, object]:
        try:
            service = get_service()
            return service.explain(data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/audit")
    def audit(payload: AuditRequest) -> dict[str, object]:
        try:
            service = get_service()
            result = run_prediction_audit(
                service,
                patient_id=payload.patient_id,
                payload=payload.payload,
                current_glucose=payload.current_glucose,
                use_gemini=payload.use_gemini,
                model=payload.model,
                timeout_seconds=payload.timeout_seconds,
            )
            return jsonable_encoder(result)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/report")
    def default_report() -> dict[str, object]:
        try:
            return get_service().get_report()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/report/{patient_id}")
    def report(patient_id: str) -> dict[str, object]:
        service = get_service()
        if patient_id not in service.records:
            raise HTTPException(status_code=404, detail=f"Unknown patient_id: {patient_id}")
        try:
            return service.get_report(patient_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/train")
    def train(payload: TrainRequest) -> dict[str, object]:
        service = get_service()
        try:
            return service.retrain(patient_ids=payload.patient_ids, persist=payload.persist)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/ingest")
    async def ingest(file: UploadFile = File(...)) -> dict[str, object]:
        service = get_service()
        raw_bytes = await file.read()
        try:
            return service.ingest_csv(raw_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/ingest-bundle")
    def ingest_bundle(payload: BundleIngestRequest) -> dict[str, object]:
        service = get_service()
        try:
            return service.ingest_bundle(
                bundle_dir=payload.bundle_dir,
                patient_id=payload.patient_id,
                retrain=payload.retrain,
                persist=payload.persist,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/ingest-ohio")
    def ingest_ohio(payload: OhioIngestRequest) -> dict[str, object]:
        service = get_service()
        try:
            return service.ingest_ohio(
                root_dir=payload.root_dir,
                split=payload.split,
                patient_ids=payload.patient_ids,
                prefix=payload.prefix,
                retrain=payload.retrain,
                persist=payload.persist,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/federated/status")
    def fl_status() -> dict[str, object]:
        return {"status": federated_status()}

    @app.post("/federated/run")
    def run_federated(payload: FederatedRunRequest) -> dict[str, object]:
        service = get_service()
        try:
            return service.run_federated_demo(
                patient_ids=payload.patient_ids,
                rounds=payload.rounds,
                min_clients=payload.min_clients,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/benchmark/ohio")
    def benchmark_ohio(payload: OhioBenchmarkRequest) -> dict[str, object]:
        service = get_service()
        try:
            return service.benchmark_ohio(
                root_dir=payload.root_dir,
                persist=payload.persist,
                max_forecast_points=payload.max_forecast_points,
                train_patient_ids=payload.train_patient_ids,
                test_patient_ids=payload.test_patient_ids,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/replay/start", response_model=ReplayStepResponse)
    def replay_start(payload: ReplayStartRequest) -> dict[str, object]:
        service = get_service()
        try:
            return service.start_replay(patient_id=payload.patient_id, start_cursor=payload.start_cursor)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/replay/step/{session_id}", response_model=ReplayStepResponse)
    def replay_step(session_id: str) -> dict[str, object]:
        service = get_service()
        try:
            return service.step_replay(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/watch/payload", response_model=WatchPayloadResponse)
    def watch_payload(
        patient_id: str | None = Query(default=None),
        session_id: str | None = Query(default=None),
    ) -> dict[str, object]:
        service = get_service()
        try:
            if session_id:
                return service.step_replay(session_id)["watch"]
            return service.get_watch_payload(patient_id=patient_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/watch", response_class=HTMLResponse)
    def watch_view(
        patient_id: str | None = Query(default=None),
        session_id: str | None = Query(default=None),
        refresh_seconds: int = Query(default=15, ge=3, le=60),
    ) -> str:
        patient_fragment = f"patient_id={patient_id}" if patient_id else ""
        session_fragment = f"session_id={session_id}" if session_id else ""
        query = "&".join(fragment for fragment in (patient_fragment, session_fragment) if fragment)
        payload_url = "/watch/payload" if not query else f"/watch/payload?{query}"
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no" />
            <title>GlycoGuard Watch</title>
            <style>
                :root {{
                    --bg: #050816;
                    --panel: #0d1323;
                    --text: #f8fafc;
                    --muted: #94a3b8;
                    --high: #ef4444;
                    --medium: #f59e0b;
                    --low: #10b981;
                }}
                * {{ box-sizing: border-box; }}
                body {{
                    margin: 0;
                    min-height: 100vh;
                    background: radial-gradient(circle at top, #111a30 0%, var(--bg) 70%);
                    color: var(--text);
                    font-family: "Segoe UI", sans-serif;
                    display: grid;
                    place-items: center;
                }}
                .watch {{
                    width: min(92vw, 360px);
                    min-height: min(92vw, 360px);
                    border-radius: 50%;
                    background: linear-gradient(180deg, rgba(13,19,35,0.96), rgba(5,8,22,0.96));
                    box-shadow: 0 22px 60px rgba(0,0,0,0.45), inset 0 0 0 1px rgba(255,255,255,0.08);
                    padding: 28px 22px;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                    text-align: center;
                }}
                .risk {{
                    display: inline-block;
                    margin: 0 auto;
                    padding: 6px 12px;
                    border-radius: 999px;
                    font-size: 12px;
                    letter-spacing: 0.08em;
                    font-weight: 700;
                    text-transform: uppercase;
                    background: rgba(148,163,184,0.14);
                }}
                .glucose {{
                    font-size: 56px;
                    line-height: 1;
                    font-weight: 700;
                    letter-spacing: -0.04em;
                }}
                .trend, .reason, .time {{
                    color: var(--muted);
                    font-size: 14px;
                }}
                .forecast {{
                    font-size: 18px;
                    font-weight: 600;
                }}
            </style>
        </head>
        <body>
            <main class="watch">
                <div>
                    <div id="risk" class="risk">Loading</div>
                    <div id="glucose" class="glucose">--</div>
                    <div id="trend" class="trend">Fetching latest risk...</div>
                </div>
                <div>
                    <div id="forecast" class="forecast">30 min: --</div>
                    <div id="reason" class="reason">Waiting for VPS prediction</div>
                    <div id="time" class="time">--</div>
                </div>
            </main>
            <script>
                const refreshMs = {refresh_seconds} * 1000;
                const riskEl = document.getElementById("risk");
                const glucoseEl = document.getElementById("glucose");
                const trendEl = document.getElementById("trend");
                const forecastEl = document.getElementById("forecast");
                const reasonEl = document.getElementById("reason");
                const timeEl = document.getElementById("time");

                function colorForRisk(risk) {{
                    if (risk === "HIGH") return "var(--high)";
                    if (risk === "MEDIUM") return "var(--medium)";
                    if (risk === "UNKNOWN") return "rgba(148,163,184,0.25)";
                    return "var(--low)";
                }}

                async function refreshPayload() {{
                    const response = await fetch("{payload_url}", {{ cache: "no-store" }});
                    const payload = await response.json();
                    if (!response.ok) {{
                        throw new Error(payload.detail || "Unable to load watch payload.");
                    }}
                    riskEl.textContent = payload.risk + " RISK";
                    riskEl.style.background = colorForRisk(payload.risk);
                    glucoseEl.textContent = `${{payload.glucose.toFixed(0)}} mg/dL`;
                    trendEl.textContent = payload.trend;
                    forecastEl.textContent = payload.forecast_30min === null ? "30 min: unknown" : `30 min: ${{payload.forecast_30min.toFixed(0)}} mg/dL`;
                    reasonEl.textContent = payload.reason;
                    timeEl.textContent = new Date(payload.updated_at).toLocaleTimeString([], {{ hour: "2-digit", minute: "2-digit" }});
                    if (payload.buzz && navigator.vibrate) {{
                        navigator.vibrate([180, 120, 180]);
                    }}
                    if (payload.status === "completed" && window.__watchTimer) {{
                        clearInterval(window.__watchTimer);
                    }}
                }}

                refreshPayload().catch((error) => {{
                    reasonEl.textContent = error.message;
                }});
                window.__watchTimer = setInterval(() => {{
                    refreshPayload().catch((error) => {{
                        reasonEl.textContent = error.message;
                    }});
                }}, refreshMs);
            </script>
        </body>
        </html>
        """

    @app.post("/artifacts/save")
    def save_artifacts(payload: ArtifactRequest) -> dict[str, object]:
        service = get_service()
        return service.save_artifacts(directory=payload.directory)

    @app.post("/artifacts/load")
    def load_artifacts(payload: ArtifactRequest) -> dict[str, object]:
        service = get_service()
        try:
            return service.load_artifacts(directory=payload.directory)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


app = create_app()
