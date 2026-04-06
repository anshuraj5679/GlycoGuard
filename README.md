# GlycoGuard

GlycoGuard is a context-aware hypoglycaemia risk intelligence system for insulin-treated diabetes. This repository bootstraps a working MVP around the build spec in your prompt:

- a causal CGM + context ingestion pipeline
- rolling clinical feature engineering
- a strict xgboost + TFT ensemble with patient-level calibration
- no synthetic startup, no fallback predictors, and no heuristic explanations in production mode
- out-of-distribution rejection and abstention instead of forced guesses
- per-prediction SHAP explanations
- a FastAPI backend and Streamlit dashboard
- OhioT1DM benchmark ingestion and replay support
- a smartwatch-safe thin-client payload and minimal watch UI
- patient profile setup plus simple meal and insulin logging endpoints

The current default configuration is strict production mode. The app does not serve predictions until real patient data has been ingested and the calibrated artifacts have been trained.

## Important environment note

The local machine currently reports `Python 3.14.0`. Several libraries in the original spec, especially `xgboost`, `shap`, `darts`, and some CGM-specific packages, may not support Python 3.14 yet. For the full research stack, use Python 3.10 or 3.11.

## Quick start

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python -m glycoguard.cli serve
```

On Windows, there is also a repo-root wrapper:

```powershell
.\run.ps1
```

This launches both:
- FastAPI at `http://127.0.0.1:8000`
- Streamlit at `http://127.0.0.1:8501`

Before requesting predictions, load real data and train:

```powershell
python -m glycoguard.cli benchmark-ohio data\raw\OhioT1DM --max-forecast-points 250
```

## API endpoints

- `GET /health`
- `GET /patients`
- `GET /profile/{patient_id}`
- `POST /profile/{patient_id}`
- `GET /report`
- `GET /report/{patient_id}`
- `POST /predict`
- `POST /explain`
- `POST /audit`
- `POST /log/meal`
- `POST /log/insulin`
- `POST /ingest`
- `POST /ingest-bundle`
- `POST /ingest-ohio`
- `POST /train`
- `GET /federated/status`
- `POST /federated/run`
- `POST /benchmark/ohio`
- `POST /replay/start`
- `POST /replay/step/{session_id}`
- `GET /watch/payload`
- `GET /watch`
- `POST /artifacts/save`
- `POST /artifacts/load`

## CLI helpers

```powershell
python -m glycoguard.cli serve
python -m glycoguard.cli patients
python -m glycoguard.cli report
python -m glycoguard.cli train --persist
python -m glycoguard.cli ingest-bundle data\raw\patient_001 --retrain
python -m glycoguard.cli benchmark-ohio data\raw\OhioT1DM --max-forecast-points 500
python -m glycoguard.cli federated-demo --rounds 3
python -m glycoguard.cli audit-prediction --patient-id bootstrap-559-ws-training --gemini
```

## Gemini audit for prediction checking

Use Gemini as a reviewer of the model output, not as the source of truth. GlycoGuard should first run its own deterministic audit against the computed prediction and only then ask Gemini to review whether the output is internally consistent.

Add your Gemini keys in the repo-root `.env`:

```powershell
GEMINI_API_KEY_1=your-first-key
GEMINI_API_KEY_2=your-second-key
GEMINI_API_KEY_3=your-third-key
GEMINI_MODEL=gemini-3.1-flash-lite
```

Then run either of these:

```powershell
python -m glycoguard.cli audit-prediction --patient-id bootstrap-559-ws-training
python -m glycoguard.cli audit-prediction --patient-id bootstrap-559-ws-training --current-glucose 92 --gemini
python -m glycoguard.cli audit-prediction --payload-file sample_payload.json --gemini
```

The command returns JSON with:
- `payload`: the exact input audited
- `prediction`: the model output from `service.explain(...)`
- `audit`: deterministic rule check for risk consistency
- `gemini_review`: optional Gemini review, with automatic fallback across `GEMINI_API_KEY_1..3`

## Android mobile client

A new Expo-based React Native client lives in [mobile/README.md](/C:/Users/Anshu%20Raj/Desktop/gluco/mobile/README.md). It renders the live watch card from `/watch/payload`, connects to your computer over LAN, and raises local Android notifications when risk escalates.

## V2 demo flows

Patient setup and low-friction logging:

```powershell
curl -X POST http://127.0.0.1:8000/ingest-ohio -H "Content-Type: application/json" -d "{\"root_dir\":\"data/raw/OhioT1DM\",\"split\":\"train\",\"prefix\":\"strict\",\"retrain\":true}"
curl http://127.0.0.1:8000/profile/strict-540-train
curl -X POST http://127.0.0.1:8000/profile/strict-540-train -H "Content-Type: application/json" -d "{\"name\":\"Priya\",\"age\":28,\"diabetes_type\":\"Type 1 diabetes\",\"insulin_therapy\":\"Bolus + basal\"}"
curl -X POST http://127.0.0.1:8000/log/meal -H "Content-Type: application/json" -d "{\"patient_id\":\"strict-540-train\",\"carb_grams\":24}"
curl -X POST http://127.0.0.1:8000/log/insulin -H "Content-Type: application/json" -d "{\"patient_id\":\"strict-540-train\",\"insulin_units\":2.5}"
```

OhioT1DM replay and watch demo:

```powershell
curl -X POST http://127.0.0.1:8000/ingest-ohio -H "Content-Type: application/json" -d "{\"root_dir\":\"data/raw/OhioT1DM\",\"split\":\"test\",\"prefix\":\"judge\"}"
curl -X POST http://127.0.0.1:8000/replay/start -H "Content-Type: application/json" -d "{\"patient_id\":\"judge-540-test\"}"
```

Then open the minimal watch interface:

```text
http://127.0.0.1:8000/watch
http://127.0.0.1:8000/watch?session_id=<your-session-id>
```

Or run the replay helper against the live API:

```powershell
python -c "from glycoguard.demo.replay import replay_session; replay_session('http://127.0.0.1:8000', patient_id='judge-540-test', speed=12)"
```

## Key implementation note

The prompt examples use `nearest` context alignment and a meal window that reaches into future timestamps. That would leak future information into training features. The implementation here uses backward-looking alignment and trailing windows so the model only sees information available at prediction time.

Strict-mode note:

- if real patient data is not loaded, `/predict`, `/explain`, `/report`, and watch endpoints return `503`
- if an input is physiologically invalid, FastAPI validation rejects it
- if an input is outside the validated training distribution, the system returns `status="insufficient_confidence"` instead of a risk guess
