from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from glycoguard.ingestion.loader import align_context, load_cgm


@dataclass(slots=True)
class PatientBundlePaths:
    cgm_path: Path
    meals_path: Optional[Path] = None
    activity_path: Optional[Path] = None
    insulin_path: Optional[Path] = None
    patient_id: Optional[str] = None


def _optional_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def load_patient_bundle(
    cgm_path: str | Path,
    meals_path: str | Path | None = None,
    activity_path: str | Path | None = None,
    insulin_path: str | Path | None = None,
    patient_id: str | None = None,
    resample_rule: str = "5min",
    interpolation_limit: int = 9,
) -> tuple[str, pd.DataFrame]:
    cgm = load_cgm(
        cgm_path,
        resample_rule=resample_rule,
        interpolation_limit=interpolation_limit,
    )
    meals = _optional_csv(Path(meals_path)) if meals_path else None
    activity = _optional_csv(Path(activity_path)) if activity_path else None
    insulin = _optional_csv(Path(insulin_path)) if insulin_path else None
    aligned = align_context(
        cgm,
        meals_df=meals,
        activity_df=activity,
        insulin_df=insulin,
        resample_rule=resample_rule,
    )
    resolved_id = patient_id or Path(cgm_path).stem
    return resolved_id, aligned


def discover_bundle_paths(bundle_dir: str | Path, patient_id: str | None = None) -> PatientBundlePaths:
    root = Path(bundle_dir)
    if not root.exists():
        raise FileNotFoundError(f"Bundle directory not found: {root}")

    candidates = {
        "cgm": ["cgm.csv", "glucose_readings.csv", "glucose.csv"],
        "meals": ["meals.csv", "food.csv"],
        "activity": ["activity.csv", "activity_sleep.csv", "wearable.csv"],
        "insulin": ["insulin.csv", "insulin_doses.csv"],
    }

    def resolve(name: str) -> Optional[Path]:
        for filename in candidates[name]:
            candidate = root / filename
            if candidate.exists():
                return candidate
        return None

    cgm_path = resolve("cgm")
    if cgm_path is None:
        raise FileNotFoundError(f"No CGM CSV found in {root}")

    return PatientBundlePaths(
        cgm_path=cgm_path,
        meals_path=resolve("meals"),
        activity_path=resolve("activity"),
        insulin_path=resolve("insulin"),
        patient_id=patient_id or root.name,
    )

