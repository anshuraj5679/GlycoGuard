from __future__ import annotations

import io
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from glycoguard.artifacts import load_json, load_pickle, save_json, save_pickle
from glycoguard.config import AppConfig, load_config
from glycoguard.evaluation import compute_binary_metrics, compute_clarke_grid, compute_lead_time
from glycoguard.explainability.shap_explainer import HypoExplainer
from glycoguard.features.engineer import FEATURE_COLUMNS, build_feature_frame_for_inference, compute_rolling_features
from glycoguard.federated.client import run_local_simulation
from glycoguard.ingestion.bundles import discover_bundle_paths, load_patient_bundle
from glycoguard.ingestion.loader import align_context, prepare_cgm_frame
from glycoguard.ingestion.ohio import load_ohio_split
from glycoguard.models.tft_model import ForecastResult, train_tft
from glycoguard.models.xgboost_model import TabularModelBundle, train_xgboost
from glycoguard.reporting import build_agp_payload, build_alert_log, build_waterfall_payload
from glycoguard.schemas import CGMInput
from glycoguard.validation import OODDetector, fit_ood_detector


@dataclass(slots=True)
class PatientRecord:
    patient_id: str
    frame: pd.DataFrame
    features: pd.DataFrame
    alert_log: list[dict[str, object]]


@dataclass(slots=True)
class PatientProfileRecord:
    patient_id: str
    name: str | None = None
    age: int | None = None
    diabetes_type: str | None = None
    insulin_therapy: str | None = None
    target_range_low: float = 70.0
    target_range_high: float = 180.0
    weight_kg: float | None = None


@dataclass(slots=True)
class ReplaySession:
    session_id: str
    patient_id: str
    cursor: int
    end_cursor: int
    status: str = "running"


@dataclass(slots=True)
class CalibrationBundle:
    calibrator: IsotonicRegression
    calibration_patient_ids: list[str]
    metrics: dict[str, float]


class GlycoGuardService:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config("configs/default.yaml")
        self.artifact_dir = Path(self.config.model.artifact_dir)
        self.records: dict[str, PatientRecord] = {}
        self.profiles: dict[str, PatientProfileRecord] = {}
        self.replay_sessions: dict[str, ReplaySession] = {}
        self.latest_benchmark: dict[str, object] | None = None
        self.latest_federated_run: dict[str, object] | None = None
        self.default_patient_id: str | None = None
        self.model_bundle: TabularModelBundle | None = None
        self.forecaster = None
        self.explainer: HypoExplainer | None = None
        self.calibration_bundle: CalibrationBundle | None = None
        self.ood_detector: OODDetector | None = None
        self._initialize_models()

    def _store_patient(self, patient_id: str, frame: pd.DataFrame) -> None:
        features = compute_rolling_features(
            frame,
            window_steps=self.config.data.window_size,
            horizon_steps=self.config.data.horizon_steps,
            hypo_threshold=self.config.data.hypo_threshold,
            severe_threshold=self.config.data.severe_threshold,
        )
        self.records[patient_id] = PatientRecord(
            patient_id=patient_id,
            frame=frame,
            features=features,
            alert_log=[],
        )
        self.profiles.setdefault(patient_id, self._default_profile(patient_id))
        if self.default_patient_id is None:
            self.default_patient_id = patient_id

    def _default_profile(self, patient_id: str) -> PatientProfileRecord:
        return PatientProfileRecord(
            patient_id=patient_id,
            name=patient_id.replace("-", " ").title(),
            diabetes_type="Insulin-treated diabetes",
            insulin_therapy="Unknown",
            target_range_low=70.0,
            target_range_high=180.0,
        )

    @staticmethod
    def _serialise_profile(profile: PatientProfileRecord) -> dict[str, object]:
        return {
            "patient_id": profile.patient_id,
            "name": profile.name,
            "age": profile.age,
            "diabetes_type": profile.diabetes_type,
            "insulin_therapy": profile.insulin_therapy,
            "target_range_low": float(profile.target_range_low),
            "target_range_high": float(profile.target_range_high),
            "weight_kg": profile.weight_kg,
        }

    def _combined_features(self, patient_ids: list[str] | None = None) -> pd.DataFrame:
        selected = patient_ids or list(self.records.keys())
        frames = [self.records[patient_id].features.copy() for patient_id in selected if patient_id in self.records]
        if not frames:
            raise ValueError("No patient feature frames are available for training.")

        combined = pd.concat(frames, axis=0)
        combined = combined.replace([np.inf, -np.inf], np.nan).dropna(subset=["hypo_label"])
        return combined

    def _representative_frame(self, patient_ids: list[str] | None = None) -> pd.DataFrame:
        selected = patient_ids or list(self.records.keys())
        candidates = [self.records[patient_id].frame for patient_id in selected if patient_id in self.records]
        if not candidates:
            raise ValueError("No patient frames are available for forecasting.")
        return max(candidates, key=len)

    def _background_frame(self, patient_ids: list[str] | None = None) -> pd.DataFrame:
        combined = self._combined_features(patient_ids=patient_ids)
        return combined.loc[:, list(FEATURE_COLUMNS)]

    def _initialize_models(self) -> None:
        if self._artifacts_available():
            try:
                self.load_artifacts()
                self._refresh_alert_logs()
                return
            except Exception:
                self.model_bundle = None
                self.forecaster = None
                self.explainer = None
                self.calibration_bundle = None
                self.ood_detector = None

    def _refresh_alert_logs(self) -> None:
        if not self.is_ready():
            return
        for record in self.records.values():
            raw_scores, _, _, valid_features = self._raw_ensemble_probabilities(
                record.frame,
                record.features,
                max_points=96,
            )
            probabilities = self.calibration_bundle.calibrator.transform(raw_scores)
            record.alert_log = build_alert_log(valid_features, probabilities)

    def _artifacts_available(self) -> bool:
        return all(
            (self.artifact_dir / name).exists()
            for name in ("model.pkl", "forecaster.pkl", "calibrator.pkl", "ood.pkl", "metadata.json")
        )

    def is_ready(self) -> bool:
        return (
            self.model_bundle is not None
            and self.forecaster is not None
            and self.explainer is not None
            and self.calibration_bundle is not None
            and self.ood_detector is not None
            and not self._artifacts_need_retrain()
        )

    def _artifacts_need_retrain(self) -> bool:
        if self.forecaster is None:
            return False
        backend = getattr(self.forecaster, "backend", "")
        forecaster_model = getattr(self.forecaster, "model", None)
        return backend == "darts_tft" and getattr(forecaster_model, "model", None) is None

    def ensure_ready(self, require_patient: bool = False) -> None:
        if not self.is_ready():
            raise RuntimeError(
                "Production mode is not ready. Load real patient data and train calibrated artifacts before requesting predictions."
            )
        if require_patient and (self.default_patient_id is None or self.default_patient_id not in self.records):
            raise RuntimeError(
                "No real patient record is loaded for report generation. Ingest OhioT1DM or a real patient bundle first."
            )

    def list_patients(self) -> list[dict[str, object]]:
        payload: list[dict[str, object]] = []
        for patient_id, record in self.records.items():
            payload.append(
                {
                    "patient_id": patient_id,
                    "rows": int(len(record.frame)),
                    "feature_rows": int(len(record.features)),
                    "start": record.frame.index.min().isoformat(),
                    "end": record.frame.index.max().isoformat(),
                    "profile": self._serialise_profile(self.profiles[patient_id]),
                }
            )
        return payload

    def get_patient_profile(self, patient_id: str) -> dict[str, object]:
        if patient_id not in self.records:
            raise ValueError(f"Unknown patient_id: {patient_id}")
        return self._serialise_profile(self.profiles[patient_id])

    def update_patient_profile(self, patient_id: str, **fields: object) -> dict[str, object]:
        if patient_id not in self.records:
            raise ValueError(f"Unknown patient_id: {patient_id}")
        profile = self.profiles.get(patient_id, self._default_profile(patient_id))
        for field, value in fields.items():
            if value is not None and hasattr(profile, field):
                setattr(profile, field, value)
        self.profiles[patient_id] = profile
        return self._serialise_profile(profile)

    def _resolve_patient_timestamp(self, patient_id: str, timestamp: pd.Timestamp | None = None) -> pd.Timestamp:
        if patient_id not in self.records:
            raise ValueError(f"Unknown patient_id: {patient_id}")
        record = self.records[patient_id]
        if timestamp is None:
            return record.frame.index[-1]
        requested = pd.Timestamp(timestamp).floor(self.config.data.resample_rule)
        if requested < record.frame.index.min() or requested > record.frame.index.max():
            raise ValueError("Timestamp must fall within the loaded patient timeline.")
        if requested in record.frame.index:
            return requested
        nearest = record.frame.index.get_indexer([requested], method="nearest")[0]
        return pd.Timestamp(record.frame.index[int(nearest)])

    def _rebuild_patient(self, patient_id: str, frame: pd.DataFrame) -> None:
        self._store_patient(patient_id, frame.sort_index())
        self._refresh_alert_logs()

    def log_meal(
        self,
        patient_id: str,
        carb_grams: float,
        timestamp: pd.Timestamp | None = None,
        description: str | None = None,
    ) -> dict[str, object]:
        ts = self._resolve_patient_timestamp(patient_id, timestamp)
        record = self.records[patient_id]
        frame = record.frame.copy()
        frame.loc[ts, "meal_carbs"] = float(frame.loc[ts, "meal_carbs"]) + float(carb_grams)
        frame.loc[(frame.index >= ts) & (frame.index <= ts + pd.Timedelta(hours=1)), "carbs_1h"] += float(carb_grams)
        frame.loc[(frame.index >= ts) & (frame.index <= ts + pd.Timedelta(hours=2)), "carbs_2h"] += float(carb_grams)
        elapsed = ((frame.index - ts) / pd.Timedelta(minutes=1)).to_numpy(dtype=float)
        elapsed = np.where(elapsed >= 0.0, elapsed, np.inf)
        frame["time_since_last_meal_min"] = np.minimum(frame["time_since_last_meal_min"].to_numpy(dtype=float), elapsed)
        self._rebuild_patient(patient_id, frame)
        return {
            "patient_id": patient_id,
            "timestamp": ts.isoformat(),
            "carb_grams": float(carb_grams),
            "description": description,
        }

    def log_insulin(
        self,
        patient_id: str,
        insulin_units: float,
        timestamp: pd.Timestamp | None = None,
        insulin_type: str = "bolus",
    ) -> dict[str, object]:
        ts = self._resolve_patient_timestamp(patient_id, timestamp)
        record = self.records[patient_id]
        frame = record.frame.copy()
        frame.loc[ts, "insulin_units"] = float(frame.loc[ts, "insulin_units"]) + float(insulin_units)

        freq_minutes = int(pd.Timedelta(self.config.data.resample_rule).total_seconds() / 60.0)
        action_steps = max(1, int(4 * 60 / max(freq_minutes, 1)))
        decay = np.linspace(float(insulin_units), 0.0, action_steps + 1)
        start_idx = int(frame.index.get_loc(ts))
        end_idx = min(len(frame), start_idx + action_steps + 1)
        usable = end_idx - start_idx
        frame.iloc[start_idx:end_idx, frame.columns.get_loc("insulin_on_board")] += decay[:usable]

        if insulin_type.lower() == "basal":
            frame.iloc[start_idx:end_idx, frame.columns.get_loc("basal_rate")] += float(insulin_units) / 4.0

        self._rebuild_patient(patient_id, frame)
        return {
            "patient_id": patient_id,
            "timestamp": ts.isoformat(),
            "insulin_units": float(insulin_units),
            "insulin_type": insulin_type,
        }

    def _build_partitions(self, patient_ids: list[str]) -> list[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
        partitions: list[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]] = []
        for patient_id in patient_ids:
            record = self.records[patient_id]
            features = record.features.loc[:, list(FEATURE_COLUMNS)]
            labels = record.features["hypo_label"].astype(int)
            split_idx = max(10, int(len(features) * 0.7))
            X_train = features.iloc[:split_idx]
            y_train = labels.iloc[:split_idx]
            X_val = features.iloc[split_idx:]
            y_val = labels.iloc[split_idx:]
            if len(X_val) < 5:
                continue
            partitions.append((X_train, y_train, X_val, y_val))
        return partitions

    def _training_patient_ids(self, patient_ids: list[str] | None = None) -> list[str]:
        selected = patient_ids or sorted(self.records.keys())
        if len(selected) < self.config.model.minimum_real_patients:
            raise ValueError(
                f"Strict mode requires at least {self.config.model.minimum_real_patients} real patients for train/calibration."
            )
        return selected

    def _split_training_and_calibration(self, patient_ids: list[str] | None = None) -> tuple[list[str], list[str]]:
        selected = self._training_patient_ids(patient_ids)
        calibration_ids = [selected[-1]]
        training_ids = selected[:-1]
        if not training_ids:
            raise ValueError("At least one patient must remain after the held-out calibration split.")
        return training_ids, calibration_ids

    def _combine_probabilities(self, classifier_probability: float, forecast_probability: float) -> float:
        return (
            self.config.model.classifier_weight * float(classifier_probability)
            + self.config.model.forecast_weight * float(forecast_probability)
        )

    def _forecast_for_position(self, frame: pd.DataFrame, position: int) -> ForecastResult:
        history = frame.iloc[position - self.config.data.window_size + 1 : position + 1]
        context = frame.iloc[position]
        return self.forecaster.predict(
            glucose_readings=history["glucose"].tolist(),
            carbs_last_hour=float(context.get("carbs_1h", 0.0)),
            carbs_last_2h=float(context.get("carbs_2h", 0.0)),
            insulin_on_board=float(context.get("insulin_on_board", 0.0)),
            activity_level=float(context.get("activity", 0.0)),
            sleep_flag=int(context.get("sleep_flag", 0)),
            stress_score=float(context.get("stress_score", 0.0)),
        )

    def _raw_ensemble_probabilities(
        self,
        frame: pd.DataFrame,
        features: pd.DataFrame,
        max_points: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        classifier_probabilities = self.model_bundle.predict_proba(features.loc[:, list(FEATURE_COLUMNS)])
        valid_rows: list[tuple[int, pd.Timestamp]] = []
        for row_idx, timestamp in enumerate(features.index):
            position = int(frame.index.get_loc(timestamp))
            if position < self.config.data.window_size - 1:
                continue
            valid_rows.append((row_idx, pd.Timestamp(timestamp)))

        if max_points is not None and len(valid_rows) > max_points:
            picks = np.linspace(0, len(valid_rows) - 1, num=max_points, dtype=int)
            valid_rows = [valid_rows[idx] for idx in picks]

        raw_scores: list[float] = []
        forecast_probabilities: list[float] = []
        classifier_kept: list[float] = []
        valid_index: list[pd.Timestamp] = []
        for row_idx, timestamp in valid_rows:
            position = int(frame.index.get_loc(timestamp))
            forecast = self._forecast_for_position(frame, position)
            forecast_probabilities.append(float(forecast.risk_probability))
            classifier_kept.append(float(classifier_probabilities[row_idx]))
            raw_scores.append(self._combine_probabilities(float(classifier_probabilities[row_idx]), float(forecast.risk_probability)))
            valid_index.append(pd.Timestamp(timestamp))
        return (
            np.asarray(raw_scores, dtype=float),
            np.asarray(classifier_kept, dtype=float),
            np.asarray(forecast_probabilities, dtype=float),
            features.loc[valid_index].copy(),
        )

    def _fit_calibration_bundle(self, calibration_ids: list[str]) -> CalibrationBundle:
        calibration_scores: list[float] = []
        calibration_labels: list[int] = []
        for patient_id in calibration_ids:
            record = self.records[patient_id]
            raw_scores, _, _, valid_features = self._raw_ensemble_probabilities(
                record.frame,
                record.features,
                max_points=160,
            )
            calibration_scores.extend(raw_scores.tolist())
            calibration_labels.extend(valid_features["hypo_label"].astype(int).tolist())

        if len(set(calibration_labels)) < 2:
            raise ValueError("Held-out calibration patients must contain both hypo and non-hypo samples.")

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(calibration_scores, calibration_labels)
        calibrated = calibrator.transform(calibration_scores)
        metrics = compute_binary_metrics(calibration_labels, calibrated)
        return CalibrationBundle(
            calibrator=calibrator,
            calibration_patient_ids=list(calibration_ids),
            metrics=metrics,
        )

    def retrain(self, patient_ids: list[str] | None = None, persist: bool = False) -> dict[str, object]:
        training_ids, calibration_ids = self._split_training_and_calibration(patient_ids)
        training_features = self._combined_features(patient_ids=training_ids)
        representative_frame = self._representative_frame(patient_ids=training_ids)
        self.model_bundle = train_xgboost(
            training_features,
            n_splits=self.config.model.n_splits,
            gap=self.config.model.gap,
            random_state=self.config.model.random_state,
        )
        self.forecaster = train_tft(
            representative_frame,
            horizon_steps=self.config.data.horizon_steps,
            prefer_tft=self.config.model.prefer_tft,
        )
        background = self._background_frame(patient_ids=training_ids)
        self.explainer = HypoExplainer(self.model_bundle, background)
        self.ood_detector = fit_ood_detector(background, quantile=self.config.model.ood_quantile)
        self.calibration_bundle = self._fit_calibration_bundle(calibration_ids)
        self._refresh_alert_logs()

        summary = {
            "trained_patients": training_ids,
            "calibration_patients": calibration_ids,
            "num_patients": len(training_ids) + len(calibration_ids),
            "num_samples": int(len(training_features)),
            "model_backend": self.model_bundle.backend,
            "forecast_backend": self.forecaster.backend,
            "metrics": self.model_bundle.metrics,
            "calibration_metrics": self.calibration_bundle.metrics,
            "ood_threshold": float(self.ood_detector.threshold),
        }
        if persist:
            self.save_artifacts()
            summary["artifact_dir"] = str(self.artifact_dir)
        return summary

    def save_artifacts(self, directory: str | Path | None = None) -> dict[str, object]:
        self.ensure_ready()
        target_dir = Path(directory) if directory is not None else self.artifact_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        save_pickle(target_dir / "model.pkl", self.model_bundle)
        save_pickle(target_dir / "forecaster.pkl", self.forecaster)
        save_pickle(target_dir / "calibrator.pkl", self.calibration_bundle)
        save_pickle(target_dir / "ood.pkl", self.ood_detector)
        save_json(
            target_dir / "metadata.json",
            {
                "default_patient_id": self.default_patient_id,
                "patients": self.list_patients(),
                "profiles": {patient_id: self._serialise_profile(profile) for patient_id, profile in self.profiles.items()},
                "metrics": self.model_bundle.metrics,
                "calibration_metrics": self.calibration_bundle.metrics,
                "model_backend": self.model_bundle.backend,
                "forecast_backend": self.forecaster.backend,
                "strict_mode": self.config.model.strict_mode,
            },
        )
        return {"artifact_dir": str(target_dir)}

    def load_artifacts(self, directory: str | Path | None = None) -> dict[str, object]:
        target_dir = Path(directory) if directory is not None else self.artifact_dir
        self.model_bundle = load_pickle(target_dir / "model.pkl")
        self.forecaster = load_pickle(target_dir / "forecaster.pkl")
        self.calibration_bundle = load_pickle(target_dir / "calibrator.pkl")
        self.ood_detector = load_pickle(target_dir / "ood.pkl")
        metadata = load_json(target_dir / "metadata.json")
        self.default_patient_id = metadata.get("default_patient_id")
        for patient_id, payload in metadata.get("profiles", {}).items():
            self.profiles[patient_id] = PatientProfileRecord(
                patient_id=patient_id,
                name=payload.get("name"),
                age=payload.get("age"),
                diabetes_type=payload.get("diabetes_type"),
                insulin_therapy=payload.get("insulin_therapy"),
                target_range_low=float(payload.get("target_range_low", 70.0)),
                target_range_high=float(payload.get("target_range_high", 180.0)),
                weight_kg=payload.get("weight_kg"),
            )
        if self.records:
            self.explainer = HypoExplainer(self.model_bundle, self._background_frame())
        return metadata

    def _sample_forecasts(
        self,
        frame: pd.DataFrame,
        feature_index: pd.Index,
        max_points: int,
    ) -> tuple[list[float], list[float]]:
        if feature_index.empty:
            return [], []

        positions = [int(frame.index.get_loc(timestamp)) for timestamp in feature_index if timestamp in frame.index]
        valid_positions = [pos for pos in positions if pos >= 23 and pos + self.config.data.horizon_steps < len(frame)]
        if not valid_positions:
            return [], []

        if len(valid_positions) > max_points:
            pick = np.linspace(0, len(valid_positions) - 1, num=max_points, dtype=int)
            valid_positions = [valid_positions[idx] for idx in pick]

        predicted: list[float] = []
        actual: list[float] = []
        for pos in valid_positions:
            history = frame.iloc[pos - 23 : pos + 1]
            context = frame.iloc[pos]
            forecast = self.forecaster.predict(
                glucose_readings=history["glucose"].tolist(),
                carbs_last_hour=float(context.get("carbs_1h", 0.0)),
                carbs_last_2h=float(context.get("carbs_2h", 0.0)),
                insulin_on_board=float(context.get("insulin_on_board", 0.0)),
                activity_level=float(context.get("activity", 0.0)),
                sleep_flag=int(context.get("sleep_flag", 0)),
                stress_score=float(context.get("stress_score", 0.0)),
            )
            predicted.append(float(forecast.forecast[-1]))
            actual.append(float(frame["glucose"].iloc[pos + self.config.data.horizon_steps]))
        return predicted, actual

    def benchmark_ohio(
        self,
        root_dir: str | Path,
        persist: bool = False,
        max_forecast_points: int = 500,
        train_patient_ids: list[str] | None = None,
        test_patient_ids: list[str] | None = None,
    ) -> dict[str, object]:
        train_filter = set(train_patient_ids) if train_patient_ids else None
        test_filter = set(test_patient_ids) if test_patient_ids else None
        train_frames = load_ohio_split(
            root_dir=root_dir,
            split="train",
            resample_rule=self.config.data.resample_rule,
            interpolation_limit=self.config.data.interpolation_limit,
            patient_ids=train_filter,
        )
        test_frames = load_ohio_split(
            root_dir=root_dir,
            split="test",
            resample_rule=self.config.data.resample_rule,
            interpolation_limit=self.config.data.interpolation_limit,
            patient_ids=test_filter,
        )

        original_records = self.records.copy()
        original_profiles = self.profiles.copy()
        original_default_patient_id = self.default_patient_id
        original_model_bundle = self.model_bundle
        original_forecaster = self.forecaster
        original_explainer = self.explainer
        original_calibration_bundle = self.calibration_bundle
        original_ood_detector = self.ood_detector
        for patient_id, frame in train_frames.items():
            self._store_patient(f"ohio-{patient_id}-train", frame)
        training_ids = [patient_id for patient_id in self.records if patient_id.startswith("ohio-") and patient_id.endswith("-train")]
        train_summary = self.retrain(patient_ids=training_ids, persist=persist)

        per_patient: list[dict[str, object]] = []
        all_truth: list[int] = []
        all_probabilities: list[float] = []
        forecast_preds: list[float] = []
        forecast_actuals: list[float] = []
        combined_leads: list[float] = []
        total_events = 0
        covered_events = 0

        for patient_id, frame in test_frames.items():
            features = compute_rolling_features(
                frame,
                window_steps=self.config.data.window_size,
                horizon_steps=self.config.data.horizon_steps,
                hypo_threshold=self.config.data.hypo_threshold,
                severe_threshold=self.config.data.severe_threshold,
            )
            raw_scores, _, _, valid_features = self._raw_ensemble_probabilities(
                frame,
                features,
                max_points=max(25, max_forecast_points // max(1, len(test_frames))),
            )
            probabilities = self.calibration_bundle.calibrator.transform(raw_scores)
            metrics = compute_binary_metrics(valid_features["hypo_label"], probabilities)
            lead_time = compute_lead_time(
                frame.loc[valid_features.index, "glucose"],
                probabilities,
                threshold=self.config.model.medium_risk_threshold,
            )
            pred_glucose, actual_glucose = self._sample_forecasts(frame, features.index, max_points=max(25, max_forecast_points // max(1, len(test_frames))))
            clarke = compute_clarke_grid(actual_glucose, pred_glucose) if actual_glucose else {"counts": {}, "percentages": {}, "zone_ab": 0.0}
            per_patient.append(
                {
                    "patient_id": patient_id,
                    "num_samples": int(len(valid_features)),
                    "binary_metrics": metrics,
                    "lead_time": lead_time,
                    "clarke_grid": clarke,
                }
            )
            all_truth.extend(valid_features["hypo_label"].astype(int).tolist())
            all_probabilities.extend([float(value) for value in probabilities])
            forecast_preds.extend(pred_glucose)
            forecast_actuals.extend(actual_glucose)
            combined_leads.extend(lead_time["lead_times"])
            total_events += int(lead_time["num_events"])
            covered_events += int(lead_time["num_events_covered"])

        overall_metrics = compute_binary_metrics(all_truth, all_probabilities) if all_truth else {}
        lead_time = {
            "num_events": total_events,
            "num_events_covered": covered_events,
            "coverage": float(covered_events / total_events) if total_events else 0.0,
            "mean_minutes": float(np.mean(combined_leads)) if combined_leads else 0.0,
            "median_minutes": float(np.median(combined_leads)) if combined_leads else 0.0,
            "max_minutes": float(np.max(combined_leads)) if combined_leads else 0.0,
            "lead_times": [float(value) for value in combined_leads],
        }
        clarke = compute_clarke_grid(forecast_actuals, forecast_preds) if forecast_actuals else {"counts": {}, "percentages": {}, "zone_ab": 0.0}

        benchmark = {
            "dataset": "OhioT1DM",
            "root_dir": str(root_dir),
            "train_patients": sorted(train_frames.keys()),
            "test_patients": sorted(test_frames.keys()),
            "train_summary": train_summary,
            "overall_metrics": overall_metrics,
            "lead_time": lead_time,
            "clarke_grid": clarke,
            "per_patient": per_patient,
            "max_forecast_points": int(max_forecast_points),
        }
        self.latest_benchmark = benchmark
        self.records = original_records
        self.profiles = original_profiles
        self.default_patient_id = original_default_patient_id
        self.model_bundle = original_model_bundle
        self.forecaster = original_forecaster
        self.explainer = original_explainer
        self.calibration_bundle = original_calibration_bundle
        self.ood_detector = original_ood_detector
        return benchmark

    def run_federated_demo(
        self,
        patient_ids: list[str] | None = None,
        rounds: int = 3,
        min_clients: int = 2,
    ) -> dict[str, object]:
        selected = patient_ids or list(self.records.keys())
        if len(selected) < min_clients:
            raise ValueError(f"At least {min_clients} patients are required for federated simulation.")

        partitions = self._build_partitions(selected[:max(min_clients, len(selected))])
        if len(partitions) < min_clients:
            raise ValueError("Not enough patient partitions with validation rows are available for federated simulation.")

        result = run_local_simulation(
            patient_partitions=partitions,
            rounds=rounds,
            random_state=self.config.model.random_state,
        )
        result["patient_ids"] = selected[: len(partitions)]
        self.latest_federated_run = result
        return result

    def _abstain_response(
        self,
        patient_id: str,
        reason: str,
        model_backend: str | None = None,
        forecast_backend: str | None = None,
        confidence: float | None = None,
    ) -> dict[str, object]:
        return {
            "patient_id": patient_id,
            "status": "insufficient_confidence",
            "hypo_probability": None,
            "classifier_probability": None,
            "forecast_probability": None,
            "risk_level": None,
            "predicted_glucose_30min": None,
            "forecast_trace": [],
            "forecast_lower": [],
            "forecast_upper": [],
            "alert_required": False,
            "model_backend": model_backend or (self.model_bundle.backend if self.model_bundle is not None else "unavailable"),
            "forecast_backend": forecast_backend or (getattr(self.forecaster, "backend", "unavailable") if self.forecaster is not None else "unavailable"),
            "confidence": None if confidence is None else round(float(confidence), 4),
            "abstention_reason": reason,
        }

    def predict(self, payload: CGMInput) -> dict[str, object]:
        self.ensure_ready()
        timestamp = pd.Timestamp(payload.timestamp) if payload.timestamp else pd.Timestamp.now().floor("5min")
        inference_features = build_feature_frame_for_inference(
            glucose_readings=payload.glucose_readings,
            carbs_last_hour=payload.carbs_last_hour,
            carbs_last_2h=payload.carbs_last_2h,
            insulin_on_board=payload.insulin_on_board,
            activity_level=payload.activity_level,
            sleep_flag=payload.sleep_flag,
            stress_score=payload.stress_score,
            timestamp=timestamp,
        )
        accepted, distances = self.ood_detector.classify(inference_features)
        patient_id = payload.patient_id or self.default_patient_id or "unknown"
        if not bool(accepted[0]):
            confidence = max(0.0, 1.0 - float(distances[0]) / max(float(self.ood_detector.threshold), 1e-6))
            return self._abstain_response(
                patient_id=patient_id,
                reason="Input lies outside the validated training distribution.",
                confidence=confidence,
            )

        classifier_probability = float(self.model_bundle.predict_proba(inference_features)[0])
        forecast: ForecastResult = self.forecaster.predict(
            glucose_readings=payload.glucose_readings,
            carbs_last_hour=payload.carbs_last_hour,
            carbs_last_2h=payload.carbs_last_hour if payload.carbs_last_2h is None else payload.carbs_last_2h,
            insulin_on_board=payload.insulin_on_board,
            activity_level=payload.activity_level,
            sleep_flag=payload.sleep_flag,
            stress_score=payload.stress_score,
        )
        raw_probability = self._combine_probabilities(classifier_probability, forecast.risk_probability)
        hypo_probability = float(self.calibration_bundle.calibrator.transform([raw_probability])[0])
        risk_level = (
            "HIGH"
            if hypo_probability >= self.config.model.high_risk_threshold
            else "MEDIUM"
            if hypo_probability >= self.config.model.medium_risk_threshold
            else "LOW"
        )
        confidence = max(0.0, 1.0 - float(distances[0]) / max(float(self.ood_detector.threshold), 1e-6))
        return {
            "patient_id": patient_id,
            "status": "ok",
            "hypo_probability": round(float(hypo_probability), 4),
            "classifier_probability": round(classifier_probability, 4),
            "forecast_probability": round(float(forecast.risk_probability), 4),
            "risk_level": risk_level,
            "predicted_glucose_30min": round(float(forecast.forecast[-1]), 1),
            "forecast_trace": [round(float(value), 1) for value in forecast.forecast],
            "forecast_lower": [round(float(value), 1) for value in forecast.lower],
            "forecast_upper": [round(float(value), 1) for value in forecast.upper],
            "alert_required": risk_level == "HIGH",
            "model_backend": self.model_bundle.backend,
            "forecast_backend": forecast.backend,
            "confidence": round(confidence, 4),
            "abstention_reason": None,
            "feature_frame": inference_features,
        }

    def explain(self, payload: CGMInput) -> dict[str, object]:
        prediction = self.predict(payload)
        if prediction["status"] != "ok":
            prediction.update(
                {
                    "explanation": prediction["abstention_reason"],
                    "top_factors": [],
                    "shap_values": {},
                    "waterfall": None,
                }
            )
            return prediction
        explanation = self.explainer.explain(prediction["feature_frame"])
        prediction.update(explanation)
        prediction.pop("feature_frame", None)
        return prediction

    def get_report(self, patient_id: str | None = None) -> dict[str, object]:
        self.ensure_ready(require_patient=True)
        pid = patient_id or self.default_patient_id
        record = self.records[pid]
        recent = record.frame.tail(24 * 12).copy()
        latest = recent.iloc[-1]
        roc_15 = float(recent["glucose"].diff(3).iloc[-1]) if len(recent) >= 4 else 0.0
        current_payload = CGMInput(
            patient_id=pid,
            glucose_readings=recent["glucose"].tail(24).tolist(),
            carbs_last_hour=float(latest.get("carbs_1h", 0.0)),
            carbs_last_2h=float(latest.get("carbs_2h", 0.0)),
            insulin_on_board=float(latest.get("insulin_on_board", 0.0)),
            activity_level=float(latest.get("activity", 0.0)),
            sleep_flag=int(latest.get("sleep_flag", 0)),
            stress_score=float(latest.get("stress_score", 0.0)),
            timestamp=recent.index[-1].to_pydatetime(),
        )
        prediction = self.explain(current_payload)
        agp = build_agp_payload(record.frame)
        watch = self.build_watch_payload(
            patient_id=pid,
            prediction=prediction,
            current_glucose=float(latest["glucose"]),
            roc_15=roc_15,
            timestamp=recent.index[-1],
        )
        return {
            "patient_id": pid,
            "profile": self._serialise_profile(self.profiles[pid]),
            "current_glucose": round(float(latest["glucose"]), 1),
            "roc_15": round(roc_15, 1),
            "prediction": prediction,
            "recent_trace": [
                {
                    "timestamp": ts.isoformat(),
                    "glucose": round(float(value), 1),
                }
                for ts, value in recent["glucose"].items()
            ],
            "context": {
                "carbs_1h": round(float(latest.get("carbs_1h", 0.0)), 1),
                "carbs_2h": round(float(latest.get("carbs_2h", 0.0)), 1),
                "insulin_on_board": round(float(latest.get("insulin_on_board", 0.0)), 2),
                "activity": round(float(latest.get("activity", 0.0)), 2),
                "sleep_flag": int(latest.get("sleep_flag", 0)),
                "stress_score": round(float(latest.get("stress_score", 0.0)), 2),
            },
            "agp": agp,
            "metrics": self.model_bundle.metrics,
            "alert_log": record.alert_log,
            "waterfall": build_waterfall_payload(prediction),
            "watch": watch,
            "benchmark": self.latest_benchmark,
            "federated": self.latest_federated_run,
        }

    def ingest_csv(self, raw_bytes: bytes, patient_id: str | None = None) -> dict[str, object]:
        frame = pd.read_csv(io.BytesIO(raw_bytes))
        cgm = prepare_cgm_frame(
            frame,
            resample_rule=self.config.data.resample_rule,
            interpolation_limit=self.config.data.interpolation_limit,
        )
        aligned = align_context(
            cgm,
            resample_rule=self.config.data.resample_rule,
        )
        pid = patient_id or f"uploaded-{uuid.uuid4().hex[:8]}"
        self._store_patient(pid, aligned)
        if self._artifacts_need_retrain() and len(self.records) >= self.config.model.minimum_real_patients:
            self.retrain(persist=False)
        elif self.model_bundle is not None and self.records:
            if self.explainer is None:
                self.explainer = HypoExplainer(self.model_bundle, self._background_frame())
            self._refresh_alert_logs()
        return {
            "patient_id": pid,
            "rows": int(len(aligned)),
            "start": aligned.index.min().isoformat(),
            "end": aligned.index.max().isoformat(),
        }

    def ingest_bundle(
        self,
        bundle_dir: str | Path,
        patient_id: str | None = None,
        retrain: bool = False,
        persist: bool = False,
    ) -> dict[str, object]:
        bundle = discover_bundle_paths(bundle_dir, patient_id=patient_id)
        resolved_id, aligned = load_patient_bundle(
            cgm_path=bundle.cgm_path,
            meals_path=bundle.meals_path,
            activity_path=bundle.activity_path,
            insulin_path=bundle.insulin_path,
            patient_id=bundle.patient_id,
            resample_rule=self.config.data.resample_rule,
            interpolation_limit=self.config.data.interpolation_limit,
        )
        self._store_patient(resolved_id, aligned)
        response = {
            "patient_id": resolved_id,
            "rows": int(len(aligned)),
            "feature_rows": int(len(self.records[resolved_id].features)),
            "start": aligned.index.min().isoformat(),
            "end": aligned.index.max().isoformat(),
        }
        if retrain or (self._artifacts_need_retrain() and len(self.records) >= self.config.model.minimum_real_patients):
            response["training"] = self.retrain(persist=persist)
        else:
            if self.model_bundle is not None and self.explainer is None:
                self.explainer = HypoExplainer(self.model_bundle, self._background_frame())
            self._refresh_alert_logs()
        return response

    def ingest_ohio(
        self,
        root_dir: str | Path,
        split: str = "test",
        patient_ids: list[str] | None = None,
        prefix: str = "ohio",
        retrain: bool = False,
        persist: bool = False,
    ) -> dict[str, object]:
        frames = load_ohio_split(
            root_dir=root_dir,
            split=split,
            resample_rule=self.config.data.resample_rule,
            interpolation_limit=self.config.data.interpolation_limit,
            patient_ids=set(patient_ids) if patient_ids else None,
        )
        loaded_ids: list[str] = []
        for source_id, frame in frames.items():
            patient_id = f"{prefix}-{source_id}-{split}"
            self._store_patient(patient_id, frame)
            self.profiles[patient_id] = PatientProfileRecord(
                patient_id=patient_id,
                name=f"OhioT1DM {source_id}",
                diabetes_type="Type 1 diabetes",
                insulin_therapy="Recorded in OhioT1DM",
                target_range_low=70.0,
                target_range_high=180.0,
            )
            loaded_ids.append(patient_id)

        response = {
            "dataset": "OhioT1DM",
            "split": split,
            "loaded_patients": loaded_ids,
            "num_patients": len(loaded_ids),
        }
        if retrain or (self._artifacts_need_retrain() and len(self.records) >= self.config.model.minimum_real_patients):
            response["training"] = self.retrain(persist=persist)
        else:
            if self.model_bundle is not None and self.explainer is None:
                self.explainer = HypoExplainer(self.model_bundle, self._background_frame())
            self._refresh_alert_logs()
        return response

    @staticmethod
    def _watch_reason(prediction: dict[str, object]) -> str:
        mapping = {
            "roc_15": "Glucose dropping fast",
            "roc_30": "Trend still falling",
            "insulin_on_board": "Active insulin still working",
            "activity": "Recent activity raising risk",
            "activity_6h": "Post-exercise low window",
            "sleep_flag": "Overnight low-risk window",
            "carbs_1h": "Recent carbs are protective",
            "time_since_last_meal_min": "Long gap since last meal",
            "min_2h": "Recent glucose already low",
            "lbgi_2h": "Pattern suggests low-glucose risk",
        }
        top_factors = prediction.get("top_factors", [])
        if prediction.get("status") != "ok":
            return str(prediction.get("abstention_reason") or "Prediction unavailable")[:48]
        if not top_factors:
            return "Stable glucose trend"
        top = top_factors[0]
        feature = str(top.get("feature", ""))
        contribution = float(top.get("contribution", 0.0))
        if prediction.get("risk_level") == "LOW" and contribution < 0:
            if feature == "carbs_1h":
                return "Recent carbs are keeping you safer"
            return "Risk is low right now"
        return mapping.get(feature, str(top.get("message", "Monitor trend")).replace("Increases risk: ", "")[:48])

    def build_watch_payload(
        self,
        patient_id: str,
        prediction: dict[str, object],
        current_glucose: float,
        roc_15: float,
        timestamp: pd.Timestamp,
        session_id: str | None = None,
        status: str = "live",
    ) -> dict[str, object]:
        return {
            "patient_id": patient_id,
            "glucose": round(float(current_glucose), 1),
            "trend": f"{float(roc_15):+.1f} mg/dL per 15min",
            "risk": prediction["risk_level"] or "UNKNOWN",
            "reason": self._watch_reason(prediction),
            "buzz": bool(prediction["alert_required"]),
            "forecast_30min": None
            if prediction.get("predicted_glucose_30min") is None
            else round(float(prediction["predicted_glucose_30min"]), 1),
            "updated_at": pd.Timestamp(timestamp).isoformat(),
            "status": status,
            "session_id": session_id,
        }

    def get_watch_payload(self, patient_id: str | None = None) -> dict[str, object]:
        report = self.get_report(patient_id=patient_id)
        timestamp = pd.Timestamp(report["recent_trace"][-1]["timestamp"])
        return self.build_watch_payload(
            patient_id=report["patient_id"],
            prediction=report["prediction"],
            current_glucose=float(report["current_glucose"]),
            roc_15=float(report["roc_15"]),
            timestamp=timestamp,
        )

    def _replay_snapshot(self, session: ReplaySession) -> dict[str, object]:
        record = self.records[session.patient_id]
        window_start = session.cursor - self.config.data.window_size + 1
        history = record.frame.iloc[window_start : session.cursor + 1]
        current = record.frame.iloc[session.cursor]
        timestamp = pd.Timestamp(record.frame.index[session.cursor])
        roc_15 = float(history["glucose"].diff(3).iloc[-1]) if len(history) >= 4 else 0.0
        payload = CGMInput(
            patient_id=session.patient_id,
            glucose_readings=history["glucose"].tolist(),
            carbs_last_hour=float(current.get("carbs_1h", 0.0)),
            carbs_last_2h=float(current.get("carbs_2h", 0.0)),
            insulin_on_board=float(current.get("insulin_on_board", 0.0)),
            activity_level=float(current.get("activity", 0.0)),
            sleep_flag=int(current.get("sleep_flag", 0)),
            stress_score=float(current.get("stress_score", 0.0)),
            timestamp=timestamp.to_pydatetime(),
        )
        prediction = self.explain(payload)
        future = record.frame["glucose"].iloc[session.cursor + 1 : session.cursor + 1 + self.config.data.horizon_steps]
        actual_glucose = float(future.iloc[-1]) if not future.empty else float(current["glucose"])
        actual_hypo = bool((future < self.config.data.hypo_threshold).any()) if not future.empty else False
        total_steps = (session.end_cursor - (self.config.data.window_size - 1)) + 1
        progress = (session.cursor - (self.config.data.window_size - 1) + 1) / max(total_steps, 1)
        return {
            "session_id": session.session_id,
            "patient_id": session.patient_id,
            "cursor": int(session.cursor),
            "total_steps": int(total_steps),
            "progress": round(float(progress), 4),
            "status": session.status,
            "timestamp": timestamp.isoformat(),
            "watch": self.build_watch_payload(
                patient_id=session.patient_id,
                prediction=prediction,
                current_glucose=float(current["glucose"]),
                roc_15=roc_15,
                timestamp=timestamp,
                session_id=session.session_id,
                status=session.status,
            ),
            "actual_glucose_30min": round(actual_glucose, 1),
            "actual_hypo": actual_hypo,
            "prediction": prediction,
        }

    def start_replay(self, patient_id: str | None = None, start_cursor: int | None = None) -> dict[str, object]:
        self.ensure_ready(require_patient=True)
        pid = patient_id or self.default_patient_id
        if pid not in self.records:
            raise ValueError(f"Unknown patient_id: {pid}")
        record = self.records[pid]
        min_cursor = self.config.data.window_size - 1
        max_cursor = len(record.frame) - self.config.data.horizon_steps - 1
        if max_cursor <= min_cursor:
            raise ValueError("Not enough rows are available to start a replay session.")
        cursor = min_cursor if start_cursor is None else max(min_cursor, min(int(start_cursor), max_cursor))
        session = ReplaySession(
            session_id=uuid.uuid4().hex[:10],
            patient_id=pid,
            cursor=cursor,
            end_cursor=max_cursor,
        )
        self.replay_sessions[session.session_id] = session
        snapshot = self._replay_snapshot(session)
        snapshot["status"] = "running"
        snapshot["watch"]["status"] = "running"
        return snapshot

    def step_replay(self, session_id: str) -> dict[str, object]:
        if session_id not in self.replay_sessions:
            raise ValueError(f"Unknown replay session: {session_id}")
        session = self.replay_sessions[session_id]
        snapshot = self._replay_snapshot(session)
        if session.cursor >= session.end_cursor:
            session.status = "completed"
            snapshot["status"] = "completed"
            snapshot["watch"]["status"] = "completed"
            return snapshot
        session.cursor += 1
        self.replay_sessions[session_id] = session
        return snapshot


@lru_cache(maxsize=1)
def get_service() -> GlycoGuardService:
    return GlycoGuardService()
