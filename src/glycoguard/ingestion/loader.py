from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def prepare_cgm_frame(
    df: pd.DataFrame,
    timestamp_column: str = "timestamp",
    glucose_column: str = "glucose",
    resample_rule: str = "5min",
    interpolation_limit: int = 9,
) -> pd.DataFrame:
    cgm = df.copy()
    cgm[timestamp_column] = pd.to_datetime(cgm[timestamp_column])
    cgm = cgm[[timestamp_column, glucose_column]].rename(columns={glucose_column: "glucose"})
    cgm = cgm.sort_values(timestamp_column).set_index(timestamp_column)
    cgm = cgm.resample(resample_rule).mean()
    cgm["glucose"] = cgm["glucose"].interpolate(
        method="time",
        limit=interpolation_limit,
        limit_area="inside",
    )
    return cgm


def load_cgm(
    path: str | Path,
    timestamp_column: str = "timestamp",
    glucose_column: str = "glucose",
    resample_rule: str = "5min",
    interpolation_limit: int = 9,
) -> pd.DataFrame:
    raw = pd.read_csv(path)
    return prepare_cgm_frame(
        raw,
        timestamp_column=timestamp_column,
        glucose_column=glucose_column,
        resample_rule=resample_rule,
        interpolation_limit=interpolation_limit,
    )


def _prepare_event_frame(
    df: Optional[pd.DataFrame],
    value_columns: list[str],
    timestamp_column: str = "timestamp",
) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    frame = df.copy()
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column])
    keep = [timestamp_column] + [column for column in value_columns if column in frame.columns]
    frame = frame[keep].sort_values(timestamp_column).set_index(timestamp_column)
    return frame


def _time_since_last_event(event_mask: pd.Series, freq_minutes: int = 5) -> pd.Series:
    elapsed = np.full(len(event_mask), np.nan, dtype=float)
    last_seen = np.nan
    for idx, has_event in enumerate(event_mask.to_numpy(dtype=bool)):
        if has_event:
            last_seen = 0.0
        elif not np.isnan(last_seen):
            last_seen += freq_minutes
        elapsed[idx] = last_seen
    return pd.Series(elapsed, index=event_mask.index).fillna(24 * 60)


def estimate_insulin_on_board(
    insulin_events: pd.Series,
    action_hours: float = 4.0,
    freq_minutes: int = 5,
) -> pd.Series:
    action_steps = max(1, int(action_hours * 60 / freq_minutes))
    decay = np.linspace(1.0, 0.0, action_steps + 1)
    iob = np.zeros(len(insulin_events), dtype=float)
    doses = insulin_events.fillna(0.0).to_numpy(dtype=float)

    for idx, dose in enumerate(doses):
        if dose <= 0:
            continue
        end_idx = min(len(iob), idx + action_steps + 1)
        usable = end_idx - idx
        iob[idx:end_idx] += dose * decay[:usable]

    return pd.Series(iob, index=insulin_events.index, name="insulin_on_board")


def _frequency_minutes(resample_rule: str) -> int:
    return max(1, int(pd.Timedelta(resample_rule).total_seconds() / 60.0))


def align_context(
    cgm_df: pd.DataFrame,
    meals_df: Optional[pd.DataFrame] = None,
    activity_df: Optional[pd.DataFrame] = None,
    insulin_df: Optional[pd.DataFrame] = None,
    resample_rule: str = "5min",
) -> pd.DataFrame:
    freq_minutes = _frequency_minutes(resample_rule)
    frame = cgm_df.copy()
    frame["meal_carbs"] = 0.0
    frame["carbs_1h"] = 0.0
    frame["carbs_2h"] = 0.0
    frame["activity"] = 0.0
    frame["sleep_flag"] = ((frame.index.hour >= 23) | (frame.index.hour < 6)).astype(int)
    frame["stress_score"] = 0.0
    frame["insulin_units"] = 0.0
    frame["basal_rate"] = 0.0
    frame["insulin_on_board"] = 0.0
    frame["time_since_last_meal_min"] = 24.0 * 60.0

    meal_events = _prepare_event_frame(meals_df, ["carb_grams"])
    if meal_events is not None:
        meal_series = meal_events["carb_grams"].resample(resample_rule).sum().reindex(frame.index, fill_value=0.0)
        frame["meal_carbs"] = meal_series
        frame["carbs_1h"] = meal_series.rolling(12, min_periods=1).sum()
        frame["carbs_2h"] = meal_series.rolling(24, min_periods=1).sum()
        frame["time_since_last_meal_min"] = _time_since_last_event(meal_series > 0.0)

    activity_events = _prepare_event_frame(activity_df, ["activity", "sleep_flag", "stress_score"])
    if activity_events is not None:
        activity_frame = activity_events.resample(resample_rule).mean().reindex(frame.index)
        frame["activity"] = activity_frame["activity"].fillna(0.0)
        if "sleep_flag" in activity_frame.columns:
            frame["sleep_flag"] = activity_frame["sleep_flag"].ffill().fillna(frame["sleep_flag"]).astype(int)
        if "stress_score" in activity_frame.columns:
            frame["stress_score"] = activity_frame["stress_score"].ffill().fillna(0.0)

    insulin_events = _prepare_event_frame(insulin_df, ["insulin_units", "insulin_on_board", "basal_rate"])
    if insulin_events is not None:
        insulin_units = insulin_events["insulin_units"].resample(resample_rule).sum().reindex(frame.index, fill_value=0.0) if "insulin_units" in insulin_events.columns else 0.0
        frame["insulin_units"] = insulin_units
        if "basal_rate" in insulin_events.columns:
            basal_rate = insulin_events["basal_rate"].resample(resample_rule).mean().reindex(frame.index).ffill().fillna(0.0)
            frame["basal_rate"] = basal_rate
            frame["insulin_units"] = frame["insulin_units"] + basal_rate * (freq_minutes / 60.0)
        insulin_frame = insulin_events.resample(resample_rule).sum().reindex(frame.index, fill_value=0.0)
        if "insulin_on_board" in insulin_frame.columns and insulin_frame["insulin_on_board"].sum() > 0:
            frame["insulin_on_board"] = insulin_frame["insulin_on_board"].ffill().fillna(0.0)
        else:
            frame["insulin_on_board"] = estimate_insulin_on_board(frame["insulin_units"], freq_minutes=freq_minutes)

    return frame
