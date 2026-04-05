from __future__ import annotations

import pytest
from conftest import write_ohio_fixture
from fastapi.testclient import TestClient

from glycoguard.api.main import app
from glycoguard.service import get_service


client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_service_cache() -> None:
    get_service.cache_clear()
    yield
    get_service.cache_clear()


def _prepare_real_service(tmp_path) -> str:
    ohio_root = write_ohio_fixture(tmp_path, patient_ids=("540", "544", "552"), rows=360)
    response = client.post(
        "/ingest-ohio",
        json={
            "root_dir": str(ohio_root),
            "split": "train",
            "prefix": "strict",
            "retrain": True,
            "persist": False,
        },
    )
    assert response.status_code == 200
    return str(ohio_root)


def test_health_and_report_fail_before_real_data() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is False
    assert body["strict_mode"] is True
    assert body["real_patient_count"] == 0

    report = client.get("/report")
    assert report.status_code == 503


def test_predict_requires_real_training_and_rejects_invalid_payload() -> None:
    payload = {
        "glucose_readings": [120 - idx for idx in range(24)],
        "carbs_last_hour": 0,
        "insulin_on_board": 2.0,
        "activity_level": 0.2,
        "sleep_flag": 1,
        "stress_score": 0.1,
    }
    not_ready = client.post("/predict", json=payload)
    assert not_ready.status_code == 503

    invalid = client.post(
        "/predict",
        json={
            "glucose_readings": [25.0] * 24,
            "carbs_last_hour": 0,
            "insulin_on_board": 2.0,
            "activity_level": 0.2,
            "sleep_flag": 1,
            "stress_score": 0.1,
        },
    )
    assert invalid.status_code == 422


def test_real_ingest_report_predict_and_watch_flow(tmp_path) -> None:
    _prepare_real_service(tmp_path)

    patients = client.get("/patients")
    assert patients.status_code == 200
    assert len(patients.json()) >= 3

    report = client.get("/report")
    assert report.status_code == 200
    report_body = report.json()
    assert report_body["prediction"]["status"] == "ok"
    assert report_body["watch"]["risk"] in {"LOW", "MEDIUM", "HIGH"}

    payload = {
        "patient_id": report_body["patient_id"],
        "glucose_readings": [118 - (idx * 1.2) for idx in range(24)],
        "carbs_last_hour": 15,
        "carbs_last_2h": 28,
        "insulin_on_board": 1.8,
        "activity_level": 0.2,
        "sleep_flag": 0,
        "stress_score": 0.2,
    }
    predict = client.post("/predict", json=payload)
    assert predict.status_code == 200
    body = predict.json()
    assert body["status"] in {"ok", "insufficient_confidence"}
    if body["status"] == "ok":
        assert 0.0 <= body["hypo_probability"] <= 1.0
        assert len(body["forecast_trace"]) == 6

    explain = client.post("/explain", json=payload)
    assert explain.status_code == 200
    assert explain.json()["status"] in {"ok", "insufficient_confidence"}

    watch_payload = client.get("/watch/payload")
    assert watch_payload.status_code == 200
    assert watch_payload.json()["risk"] in {"LOW", "MEDIUM", "HIGH", "UNKNOWN"}

    watch_page = client.get("/watch")
    assert watch_page.status_code == 200
    assert "GlycoGuard Watch" in watch_page.text


def test_profile_logging_replay_and_federated_endpoints(tmp_path) -> None:
    _prepare_real_service(tmp_path)
    patient_id = client.get("/report").json()["patient_id"]

    profile = client.post(
        f"/profile/{patient_id}",
        json={
            "name": "Priya",
            "age": 28,
            "diabetes_type": "Type 1 diabetes",
            "insulin_therapy": "Bolus + basal",
            "weight_kg": 58.0,
        },
    )
    assert profile.status_code == 200
    assert profile.json()["name"] == "Priya"

    meal = client.post("/log/meal", json={"patient_id": patient_id, "carb_grams": 20.0})
    assert meal.status_code == 200

    insulin = client.post("/log/insulin", json={"patient_id": patient_id, "insulin_units": 2.5})
    assert insulin.status_code == 200

    replay = client.post("/replay/start", json={"patient_id": patient_id})
    assert replay.status_code == 200
    session_id = replay.json()["session_id"]

    replay_watch = client.get(f"/watch/payload?session_id={session_id}")
    assert replay_watch.status_code == 200
    assert replay_watch.json()["session_id"] == session_id

    federated = client.post("/federated/run", json={"rounds": 1, "min_clients": 2})
    assert federated.status_code == 200
    assert federated.json()["status"] == "completed"


def test_benchmark_endpoint_runs_with_real_ohio_fixture(tmp_path) -> None:
    ohio_root = write_ohio_fixture(tmp_path, patient_ids=("540", "544", "552"), rows=360)
    response = client.post(
        "/benchmark/ohio",
        json={
            "root_dir": str(ohio_root),
            "persist": False,
            "max_forecast_points": 40,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["dataset"] == "OhioT1DM"
    assert "clarke_grid" in body
    assert "overall_metrics" in body
