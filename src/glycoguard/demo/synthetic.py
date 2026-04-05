from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SyntheticPatientBundle:
    patient_id: str
    cgm: pd.DataFrame
    meals: pd.DataFrame
    activity: pd.DataFrame
    insulin: pd.DataFrame


def _add_effect(signal: np.ndarray, start_idx: int, weights: np.ndarray, scale: float) -> None:
    if start_idx >= len(signal):
        return

    end_idx = min(len(signal), start_idx + len(weights))
    usable = end_idx - start_idx
    signal[start_idx:end_idx] += weights[:usable] * scale


def generate_synthetic_bundle(
    days: int = 14,
    seed: int = 42,
    patient_id: str = "demo-001",
) -> SyntheticPatientBundle:
    rng = np.random.default_rng(seed)
    index = pd.date_range(
        end=pd.Timestamp.now().floor("5min"),
        periods=days * 24 * 12,
        freq="5min",
    )
    n_rows = len(index)

    hours = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    glucose = 112.0 + 12.0 * np.sin(2.0 * np.pi * (hours - 5.0) / 24.0)
    glucose += rng.normal(0.0, 4.5, n_rows)

    activity_level = np.zeros(n_rows, dtype=float)
    sleep_flag = ((index.hour >= 23) | (index.hour < 6)).astype(int)
    stress_score = np.clip(
        0.18 + 0.10 * np.sin(2.0 * np.pi * (hours + 1.5) / 24.0) + rng.normal(0.0, 0.05, n_rows),
        0.0,
        1.0,
    )

    meal_times: list[pd.Timestamp] = []
    carb_values: list[float] = []
    insulin_times: list[pd.Timestamp] = []
    insulin_units: list[float] = []

    dates = pd.Index(index.normalize().unique())
    meal_specs = (
        (8, 0, 38, 72),
        (13, 0, 50, 95),
        (19, 0, 42, 88),
    )

    meal_rise = np.exp(-0.5 * ((np.arange(18) - 7) / 3.2) ** 2)
    insulin_drop = np.linspace(1.0, 0.0, 48)
    short_activity_drop = np.exp(-0.5 * ((np.arange(12) - 5) / 2.2) ** 2)
    delayed_activity_drop = np.exp(-0.5 * ((np.arange(72) - 30) / 12.0) ** 2)
    nocturnal_dip = np.exp(-0.5 * ((np.arange(20) - 9) / 4.0) ** 2)

    for day_idx, current_day in enumerate(dates):
        for hour, minute, low_carb, high_carb in meal_specs:
            meal_time = current_day + pd.Timedelta(hours=hour, minutes=minute)
            meal_time += pd.Timedelta(minutes=int(rng.integers(-25, 26)))
            carbs = float(rng.integers(low_carb, high_carb))
            meal_times.append(meal_time)
            carb_values.append(carbs)

            bolus = round(max(1.0, carbs / 11.0 + rng.normal(0.0, 0.6)), 2)
            insulin_times.append(meal_time - pd.Timedelta(minutes=int(rng.integers(0, 16))))
            insulin_units.append(bolus)

            meal_idx = index.get_indexer([meal_time], method="nearest")[0]
            _add_effect(glucose, meal_idx, meal_rise, carbs / 3.6)
            _add_effect(glucose, meal_idx + 2, insulin_drop, -bolus * 4.1)

        if rng.random() < 0.6:
            exercise_time = current_day + pd.Timedelta(hours=18, minutes=int(rng.integers(-35, 36)))
            exercise_idx = index.get_indexer([exercise_time], method="nearest")[0]
            duration_steps = int(rng.integers(8, 15))
            intensity = float(rng.uniform(0.45, 0.95))
            activity_level[exercise_idx : exercise_idx + duration_steps] = np.maximum(
                activity_level[exercise_idx : exercise_idx + duration_steps],
                intensity,
            )
            _add_effect(glucose, exercise_idx, short_activity_drop, -8.0 * intensity)
            _add_effect(glucose, exercise_idx + 72, delayed_activity_drop, -10.0 * intensity)

            if rng.random() < 0.55:
                dip_time = current_day + pd.Timedelta(days=1, hours=int(rng.integers(1, 4)))
                dip_idx = index.get_indexer([dip_time], method="nearest")[0]
                _add_effect(glucose, dip_idx, nocturnal_dip, -rng.uniform(18.0, 30.0))

        if day_idx % 5 == 2:
            overnight_time = current_day + pd.Timedelta(hours=2, minutes=int(rng.integers(-20, 21)))
            overnight_idx = index.get_indexer([overnight_time], method="nearest")[0]
            _add_effect(glucose, overnight_idx, nocturnal_dip, -rng.uniform(12.0, 22.0))

    glucose -= (activity_level > 0).astype(float) * 2.0
    glucose -= sleep_flag * 1.2
    glucose += stress_score * 4.0
    glucose = np.clip(glucose, 42.0, 260.0)

    cgm = pd.DataFrame({"timestamp": index, "glucose": glucose})
    meals = pd.DataFrame({"timestamp": meal_times, "carb_grams": carb_values}).sort_values("timestamp")
    activity = pd.DataFrame(
        {
            "timestamp": index,
            "activity": activity_level,
            "sleep_flag": sleep_flag.astype(int),
            "stress_score": stress_score,
        }
    )
    insulin = pd.DataFrame(
        {"timestamp": insulin_times, "insulin_units": insulin_units}
    ).sort_values("timestamp")

    return SyntheticPatientBundle(
        patient_id=patient_id,
        cgm=cgm,
        meals=meals,
        activity=activity,
        insulin=insulin,
    )

