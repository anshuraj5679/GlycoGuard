from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from contextlib import contextmanager
import logging
import warnings

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    from darts import TimeSeries
    from darts.dataprocessing.transformers import Scaler
    from darts.models.forecasting.tft_model import TFTModel

    try:
        from darts.utils.likelihood_models import QuantileRegression
    except ImportError:  # pragma: no cover - optional dependency
        from darts.utils.likelihood_models import QuantileRegressionLikelihood as QuantileRegression
    DARTS_TFT_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency
    TimeSeries = None
    Scaler = None
    TFTModel = None
    QuantileRegression = None
    DARTS_TFT_IMPORT_ERROR = exc


@dataclass(slots=True)
class ForecastResult:
    forecast: list[float]
    lower: list[float]
    upper: list[float]
    risk_probability: float
    backend: str


def minimum_tft_rows(horizon_steps: int = 6) -> int:
    return max(288, horizon_steps * 30)


def preferred_tft_training_rows(horizon_steps: int = 6) -> int:
    return max(minimum_tft_rows(horizon_steps), 864)


def tft_runtime_available() -> bool:
    return TimeSeries is not None and TFTModel is not None and Scaler is not None


def tft_runtime_error() -> str | None:
    if DARTS_TFT_IMPORT_ERROR is None:
        return None
    return f"{type(DARTS_TFT_IMPORT_ERROR).__name__}: {DARTS_TFT_IMPORT_ERROR}"


@contextmanager
def _suppress_tft_runtime_noise():
    logger_names = [
        "pytorch_lightning",
        "pytorch_lightning.utilities.rank_zero",
        "lightning",
        "lightning.pytorch",
    ]
    original_levels = {name: logging.getLogger(name).level for name in logger_names}
    for name in logger_names:
        logging.getLogger(name).setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*LeafSpec.*deprecated.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*pin_memory.*no accelerator.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*predict_dataloader.*does not have many workers.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*train_dataloader.*does not have many workers.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*val_dataloader.*does not have many workers.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*validation_step.*no `val_dataloader`.*",
            category=UserWarning,
        )
        try:
            yield
        finally:
            for name, level in original_levels.items():
                logging.getLogger(name).setLevel(level)


class DartsTFTForecaster:
    def __init__(self, horizon_steps: int = 6, random_state: int = 42) -> None:
        self.horizon_steps = horizon_steps
        self.random_state = random_state
        self.backend = "darts_tft"
        self.target_scaler = Scaler()
        with _suppress_tft_runtime_noise():
            self.model = TFTModel(
                input_chunk_length=24,
                output_chunk_length=horizon_steps,
                hidden_size=32,
                lstm_layers=1,
                num_attention_heads=4,
                dropout=0.1,
                batch_size=32,
                n_epochs=3,
                random_state=random_state,
                force_reset=True,
                likelihood=QuantileRegression([0.05, 0.5, 0.95]) if QuantileRegression is not None else None,
                pl_trainer_kwargs={
                    "accelerator": "cpu",
                    "devices": 1,
                    "enable_model_summary": False,
                    "enable_progress_bar": False,
                    "logger": False,
                },
            )
        self.training_columns = ["carbs_1h", "activity", "insulin_on_board", "sleep_flag", "stress_score"]
        self.dataloader_kwargs = {"pin_memory": False, "num_workers": 0}

    def _covariates_from_frame(self, frame: pd.DataFrame) -> "TimeSeries":
        cov_frame = frame.copy()
        for column in self.training_columns:
            if column not in cov_frame.columns:
                cov_frame[column] = 0.0
        return TimeSeries.from_dataframe(
            cov_frame.reset_index(),
            time_col="timestamp",
            value_cols=self.training_columns,
            fill_missing_dates=True,
            freq="5min",
        )

    def fit(self, df: pd.DataFrame) -> "DartsTFTForecaster":
        frame = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})
        if "timestamp" not in frame.columns:
            frame = frame.rename(columns={frame.columns[0]: "timestamp"})
        series = TimeSeries.from_dataframe(
            frame,
            time_col="timestamp",
            value_cols=["glucose"],
            fill_missing_dates=True,
            freq="5min",
        )
        cov = self._covariates_from_frame(frame.set_index("timestamp"))
        scaled = self.target_scaler.fit_transform(series)
        dataloader_kwargs = getattr(self, "dataloader_kwargs", {"pin_memory": False, "num_workers": 0})
        with _suppress_tft_runtime_noise():
            self.model.fit(
                scaled,
                future_covariates=cov,
                verbose=False,
                dataloader_kwargs=dataloader_kwargs,
            )
        return self

    def predict(
        self,
        glucose_readings: Sequence[float],
        carbs_last_hour: float = 0.0,
        carbs_last_2h: float = 0.0,
        insulin_on_board: float = 0.0,
        activity_level: float = 0.0,
        sleep_flag: int = 0,
        stress_score: float = 0.0,
    ) -> ForecastResult:
        values = list(glucose_readings)
        end = pd.Timestamp.now().floor("5min")
        past_index = pd.date_range(end=end, periods=len(values), freq="5min")
        future_index = pd.date_range(start=end + pd.Timedelta(minutes=5), periods=self.horizon_steps, freq="5min")

        target_frame = pd.DataFrame({"timestamp": past_index, "glucose": values})
        cov_index = past_index.append(future_index)
        cov_frame = pd.DataFrame(
            {
                "timestamp": cov_index,
                "carbs_1h": carbs_last_hour,
                "activity": activity_level,
                "insulin_on_board": insulin_on_board,
                "sleep_flag": sleep_flag,
                "stress_score": stress_score,
            }
        )
        series = TimeSeries.from_dataframe(
            target_frame,
            time_col="timestamp",
            value_cols=["glucose"],
            fill_missing_dates=True,
            freq="5min",
        )
        cov = TimeSeries.from_dataframe(
            cov_frame,
            time_col="timestamp",
            value_cols=self.training_columns,
            fill_missing_dates=True,
            freq="5min",
        )
        scaled_series = self.target_scaler.transform(series)
        dataloader_kwargs = getattr(self, "dataloader_kwargs", {"pin_memory": False, "num_workers": 0})
        with _suppress_tft_runtime_noise():
            prediction = self.model.predict(
                n=self.horizon_steps,
                series=scaled_series,
                future_covariates=cov,
                num_samples=100,
                dataloader_kwargs=dataloader_kwargs,
            )
        prediction = self.target_scaler.inverse_transform(prediction)
        samples = prediction.all_values(copy=False)
        median = np.quantile(samples, 0.5, axis=2).reshape(-1)
        lower = np.quantile(samples, 0.05, axis=2).reshape(-1)
        upper = np.quantile(samples, 0.95, axis=2).reshape(-1)
        risk_probability = float(np.clip(np.mean(lower < 70.0), 0.0, 1.0))
        return ForecastResult(
            forecast=[float(item) for item in median],
            lower=[float(item) for item in lower],
            upper=[float(item) for item in upper],
            risk_probability=risk_probability,
            backend=self.backend,
        )


def train_tft(df: pd.DataFrame, horizon_steps: int = 6, prefer_tft: bool = True):
    min_rows = minimum_tft_rows(horizon_steps)
    if len(df) < min_rows:
        raise ValueError(
            f"TFT training requires at least {min_rows} rows at 5-minute cadence; received {len(df)}."
        )

    if not tft_runtime_available():
        error_message = tft_runtime_error() or "unknown import error"
        raise RuntimeError(f"TFT dependencies are unavailable in strict mode: {error_message}")

    training_frame = df.tail(preferred_tft_training_rows(horizon_steps)).copy()
    try:
        return DartsTFTForecaster(horizon_steps=horizon_steps).fit(training_frame)
    except Exception as exc:
        raise RuntimeError(f"TFT training failed in strict mode: {type(exc).__name__}: {exc}") from exc
