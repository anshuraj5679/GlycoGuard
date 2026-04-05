from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd


FEATURE_COLUMNS: Sequence[str] = (
    "glucose",
    "roc_5",
    "roc_15",
    "roc_30",
    "roc_accel_15",
    "mean_30m",
    "mean_2h",
    "std_30m",
    "std_2h",
    "min_2h",
    "max_2h",
    "range_2h",
    "cv_2h",
    "pct_below_70_2h",
    "pct_below_54_2h",
    "time_in_range_2h",
    "lbgi_2h",
    "hbgi_2h",
    "glucose_deficit_2h",
    "carbs_1h",
    "carbs_2h",
    "insulin_units",
    "insulin_on_board",
    "activity",
    "activity_2h",
    "activity_6h",
    "sleep_flag",
    "stress_score",
    "hour_sin",
    "hour_cos",
    "is_night",
    "time_since_last_meal_min",
)


def _risk_transform(glucose: pd.Series) -> tuple[pd.Series, pd.Series]:
    clipped = glucose.clip(lower=18)
    f = 1.509 * ((np.log(clipped) ** 1.084) - 5.381)
    low = 10.0 * np.square(np.minimum(f, 0.0))
    high = 10.0 * np.square(np.maximum(f, 0.0))
    return pd.Series(low, index=glucose.index), pd.Series(high, index=glucose.index)


def compute_rolling_features(
    df: pd.DataFrame,
    window_steps: int = 24,
    horizon_steps: int = 6,
    hypo_threshold: float = 70.0,
    severe_threshold: float = 54.0,
) -> pd.DataFrame:
    frame = df.copy()
    feats = pd.DataFrame(index=frame.index)

    low_risk, high_risk = _risk_transform(frame["glucose"])

    feats["glucose"] = frame["glucose"]
    feats["roc_5"] = frame["glucose"].diff(1)
    feats["roc_15"] = frame["glucose"].diff(3)
    feats["roc_30"] = frame["glucose"].diff(6)
    feats["roc_accel_15"] = feats["roc_15"].diff(3)

    feats["mean_30m"] = frame["glucose"].rolling(6, min_periods=3).mean()
    feats["mean_2h"] = frame["glucose"].rolling(window_steps, min_periods=12).mean()
    feats["std_30m"] = frame["glucose"].rolling(6, min_periods=3).std()
    feats["std_2h"] = frame["glucose"].rolling(window_steps, min_periods=12).std()
    feats["min_2h"] = frame["glucose"].rolling(window_steps, min_periods=12).min()
    feats["max_2h"] = frame["glucose"].rolling(window_steps, min_periods=12).max()
    feats["range_2h"] = feats["max_2h"] - feats["min_2h"]
    feats["cv_2h"] = feats["std_2h"] / feats["mean_2h"].replace(0, np.nan)
    feats["pct_below_70_2h"] = (
        (frame["glucose"] < hypo_threshold).astype(float).rolling(window_steps, min_periods=12).mean()
    )
    feats["pct_below_54_2h"] = (
        (frame["glucose"] < severe_threshold).astype(float).rolling(window_steps, min_periods=12).mean()
    )
    feats["time_in_range_2h"] = (
        frame["glucose"].between(hypo_threshold, 180.0).astype(float).rolling(window_steps, min_periods=12).mean()
    )
    feats["lbgi_2h"] = low_risk.rolling(window_steps, min_periods=12).mean()
    feats["hbgi_2h"] = high_risk.rolling(window_steps, min_periods=12).mean()
    feats["glucose_deficit_2h"] = (
        (hypo_threshold - frame["glucose"]).clip(lower=0.0).rolling(window_steps, min_periods=12).sum()
    )

    feats["carbs_1h"] = frame.get("carbs_1h", 0.0)
    feats["carbs_2h"] = frame.get("carbs_2h", feats["carbs_1h"])
    feats["insulin_units"] = frame.get("insulin_units", 0.0)
    feats["insulin_on_board"] = frame.get("insulin_on_board", 0.0)
    feats["activity"] = frame.get("activity", 0.0)
    feats["activity_2h"] = feats["activity"].rolling(window_steps, min_periods=1).mean()
    feats["activity_6h"] = feats["activity"].rolling(window_steps * 3, min_periods=1).mean()
    feats["sleep_flag"] = frame.get("sleep_flag", 0).astype(int)
    feats["stress_score"] = frame.get("stress_score", 0.0)

    hours = frame.index.hour + frame.index.minute / 60.0
    feats["hour_sin"] = np.sin(2.0 * math.pi * hours / 24.0)
    feats["hour_cos"] = np.cos(2.0 * math.pi * hours / 24.0)
    feats["is_night"] = ((frame.index.hour >= 22) | (frame.index.hour <= 6)).astype(int)
    feats["time_since_last_meal_min"] = frame.get("time_since_last_meal_min", 24.0 * 60.0)

    future_min = frame["glucose"].shift(-1).rolling(horizon_steps, min_periods=horizon_steps).min()
    feats["hypo_label"] = (future_min < hypo_threshold).astype(int)
    feats["severe_hypo_label"] = (future_min < severe_threshold).astype(int)

    return feats.replace([np.inf, -np.inf], np.nan).dropna()


def build_feature_frame_for_inference(
    glucose_readings: Sequence[float],
    carbs_last_hour: float = 0.0,
    carbs_last_2h: float | None = None,
    insulin_on_board: float = 0.0,
    activity_level: float = 0.0,
    sleep_flag: int = 0,
    stress_score: float = 0.0,
    timestamp: pd.Timestamp | None = None,
) -> pd.DataFrame:
    readings = list(glucose_readings)
    if len(readings) < 24:
        raise ValueError("At least 24 glucose readings are required for inference.")

    ts = pd.Timestamp.now().floor("5min") if timestamp is None else pd.Timestamp(timestamp)
    index = pd.date_range(end=ts, periods=len(readings), freq="5min")
    frame = pd.DataFrame(
        {
            "glucose": readings,
            "carbs_1h": carbs_last_hour,
            "carbs_2h": carbs_last_hour if carbs_last_2h is None else carbs_last_2h,
            "insulin_units": 0.0,
            "insulin_on_board": insulin_on_board,
            "activity": activity_level,
            "sleep_flag": int(sleep_flag),
            "stress_score": stress_score,
            "time_since_last_meal_min": 60.0 if carbs_last_hour > 0 else 180.0,
        },
        index=index,
    )

    features = compute_rolling_features(frame)
    return features.iloc[[-1]].loc[:, list(FEATURE_COLUMNS)]

