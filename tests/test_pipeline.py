from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from conftest import write_ohio_fixture
from glycoguard.config import load_config
from glycoguard.features.engineer import FEATURE_COLUMNS, build_feature_frame_for_inference, compute_rolling_features
from glycoguard.ingestion.bundles import discover_bundle_paths
from glycoguard.ingestion.loader import align_context, prepare_cgm_frame
from glycoguard.ingestion.ohio import discover_ohio_split_dirs, load_ohio_split
from glycoguard.service import GlycoGuardService


def _strict_config(tmp_path: Path):
    os.environ["GLYCOGUARD_BOOTSTRAP_OHIO_DIR"] = ""
    config = load_config("configs/default.yaml")
    config.model.artifact_dir = str(tmp_path / "artifacts")
    return config


def test_feature_pipeline_has_expected_columns(tmp_path: Path) -> None:
    ohio_root = write_ohio_fixture(tmp_path, rows=360)
    frame = next(iter(load_ohio_split(ohio_root, "train").values()))
    features = compute_rolling_features(frame)
    for column in FEATURE_COLUMNS:
        assert column in features.columns
    assert "hypo_label" in features.columns
    assert len(features) > 100


def test_inference_feature_builder() -> None:
    frame = build_feature_frame_for_inference(
        glucose_readings=[110 - step for step in range(24)],
        carbs_last_hour=0.0,
        insulin_on_board=1.5,
        activity_level=0.2,
        sleep_flag=1,
        stress_score=0.1,
    )
    assert frame.shape == (1, len(FEATURE_COLUMNS))


def test_service_is_not_ready_without_real_data(tmp_path: Path) -> None:
    service = GlycoGuardService(config=_strict_config(tmp_path))
    assert service.is_ready() is False
    assert service.list_patients() == []


def test_bundle_ingestion_and_retrain(tmp_path: Path) -> None:
    timestamps = pd.date_range("2026-01-01", periods=420, freq="5min")
    cgm = pd.DataFrame({"timestamp": timestamps, "glucose": 118 + pd.Series(range(420)).mod(24).astype(float)})
    meals = pd.DataFrame({"timestamp": [timestamps[60], timestamps[180]], "carb_grams": [45.0, 30.0]})
    activity = pd.DataFrame({"timestamp": timestamps[::12], "activity": [0.2] * len(timestamps[::12])})
    insulin = pd.DataFrame({"timestamp": [timestamps[58], timestamps[178]], "insulin_units": [4.0, 3.0]})

    bundle_dir = tmp_path / "bundle-patient"
    bundle_dir.mkdir()
    cgm.to_csv(bundle_dir / "cgm.csv", index=False)
    meals.to_csv(bundle_dir / "meals.csv", index=False)
    activity.to_csv(bundle_dir / "activity_sleep.csv", index=False)
    insulin.to_csv(bundle_dir / "insulin_doses.csv", index=False)

    ohio_root = write_ohio_fixture(tmp_path / "ohio", patient_ids=("540", "544"), rows=360)
    service = GlycoGuardService(config=_strict_config(tmp_path))
    service.ingest_ohio(ohio_root, split="train", prefix="seed", retrain=False)

    discovered = discover_bundle_paths(bundle_dir)
    assert discovered.cgm_path.name == "cgm.csv"

    response = service.ingest_bundle(bundle_dir=bundle_dir, retrain=True)
    assert response["patient_id"] == "bundle-patient"
    assert response["training"]["num_patients"] >= 3
    assert service.is_ready() is True


def test_artifact_round_trip_restores_real_records(tmp_path: Path) -> None:
    ohio_root = write_ohio_fixture(tmp_path, patient_ids=("540", "544", "552"), rows=360)
    config = _strict_config(tmp_path)
    service = GlycoGuardService(config=config)
    service.ingest_ohio(ohio_root, split="train", prefix="strict", retrain=True, persist=True)

    restored = GlycoGuardService(config=config)
    assert restored.is_ready() is True
    assert len(restored.records) == 3


def test_ohio_loader_and_benchmark_flow(tmp_path: Path) -> None:
    ohio_root = write_ohio_fixture(tmp_path, patient_ids=("540", "544", "552"), rows=360)
    dirs = discover_ohio_split_dirs(ohio_root)
    assert dirs.train_dir.exists()
    assert dirs.test_dir.exists()

    train_split = load_ohio_split(ohio_root, "train")
    test_split = load_ohio_split(ohio_root, "test")
    assert train_split
    assert test_split

    service = GlycoGuardService(config=_strict_config(tmp_path))
    benchmark = service.benchmark_ohio(ohio_root, max_forecast_points=40)
    assert benchmark["dataset"] == "OhioT1DM"
    assert "overall_metrics" in benchmark
    assert "clarke_grid" in benchmark
    assert "lead_time" in benchmark


def test_federated_demo_and_waterfall_payload(tmp_path: Path) -> None:
    ohio_root = write_ohio_fixture(tmp_path, patient_ids=("540", "544", "552"), rows=360)
    service = GlycoGuardService(config=_strict_config(tmp_path))
    service.ingest_ohio(ohio_root, split="train", prefix="strict", retrain=True, persist=False)

    result = service.run_federated_demo(rounds=2, min_clients=2)
    assert result["status"] == "completed"
    assert result["num_clients"] >= 2

    report = service.get_report()
    assert "waterfall" in report
    assert report["prediction"]["status"] == "ok"


def test_logging_updates_patient_without_global_alert_refresh(tmp_path: Path) -> None:
    ohio_root = write_ohio_fixture(tmp_path, patient_ids=("540", "544", "552"), rows=360)
    service = GlycoGuardService(config=_strict_config(tmp_path))
    service.ingest_ohio(ohio_root, split="train", prefix="strict", retrain=True, persist=False)

    patient_id = service.default_patient_id
    assert patient_id is not None
    prior_alert_log = [{"timestamp": "seed", "risk_level": "LOW"}]
    service.records[patient_id].alert_log = prior_alert_log.copy()

    def fail_refresh() -> None:
        raise AssertionError("global alert refresh should not run during meal/insulin logging")

    service._refresh_alert_logs = fail_refresh  # type: ignore[method-assign]

    meal_result = service.log_meal(patient_id, carb_grams=15.0)
    insulin_result = service.log_insulin(patient_id, insulin_units=2.0, insulin_type="bolus")

    assert meal_result["patient_id"] == patient_id
    assert insulin_result["patient_id"] == patient_id
    assert service.records[patient_id].alert_log == prior_alert_log


def test_report_clamps_loaded_context_before_building_cgm_input(tmp_path: Path) -> None:
    ohio_root = write_ohio_fixture(tmp_path, patient_ids=("540", "544", "552"), rows=360)
    service = GlycoGuardService(config=_strict_config(tmp_path))
    service.ingest_ohio(ohio_root, split="train", prefix="strict", retrain=True, persist=False)

    patient_id = service.default_patient_id
    assert patient_id is not None

    for _ in range(3):
        service.log_insulin(patient_id, insulin_units=10.0, insulin_type="bolus")

    report = service.get_report()

    assert report["patient_id"] == patient_id
    assert report["context"]["insulin_on_board"] == 25.0
    assert report["prediction"]["status"] in {"ok", "insufficient_confidence"}


def test_predict_abstains_when_forecaster_runtime_fails(tmp_path: Path) -> None:
    ohio_root = write_ohio_fixture(tmp_path, patient_ids=("540", "544", "552"), rows=360)
    service = GlycoGuardService(config=_strict_config(tmp_path))
    service.ingest_ohio(ohio_root, split="train", prefix="strict", retrain=True, persist=False)

    def broken_predict(**_: object):
        raise RuntimeError("synthetic TFT failure")

    service.forecaster.predict = broken_predict  # type: ignore[method-assign]
    report = service.get_report()

    assert report["prediction"]["status"] == "insufficient_confidence"
    assert "Forecast backend failed at runtime" in report["prediction"]["abstention_reason"]


def test_benchmark_failure_restores_live_service_state(tmp_path: Path) -> None:
    ohio_root = write_ohio_fixture(tmp_path, patient_ids=("540", "544", "552"), rows=360)
    service = GlycoGuardService(config=_strict_config(tmp_path))
    service.ingest_ohio(ohio_root, split="train", prefix="strict", retrain=True, persist=False)
    original_patient_id = service.default_patient_id

    def broken_sample(*args, **kwargs):
        raise RuntimeError("benchmark forecast failure")

    service._sample_forecasts = broken_sample  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        service.benchmark_ohio(ohio_root, max_forecast_points=20)

    assert service.default_patient_id == original_patient_id
    assert service.is_ready() is True
    live_report = service.get_report()
    assert live_report["patient_id"] == original_patient_id


def test_prepare_cgm_and_align_context_stay_causal() -> None:
    timestamps = pd.date_range("2026-01-01", periods=30, freq="7min")
    raw = pd.DataFrame({"timestamp": timestamps, "glucose": [110.0] * len(timestamps)})
    cgm = prepare_cgm_frame(raw)
    aligned = align_context(cgm)
    assert "insulin_on_board" in aligned.columns
    assert "carbs_1h" in aligned.columns
    assert aligned.index.is_monotonic_increasing
